import numpy as np
import tensorflow as tf
from tensorflow import keras
from model.custom_op.op_interfaces import ball_grouping, fps_sampling, knn_grouping, interpolate


class SharedDense(keras.layers.Layer):
    def __init__(self, units, use_bias=True, use_bn=False, activation=None, kernel_initializer="GlorotUniform", bias_initializer="zeros", **kwargs):
        super(SharedDense, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias and not use_bn
        self.use_bn = use_bn
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        # shared linear, i.e. Conv
        if len(input_shape) == 3:  # Conv1D, [N,W,C]
            self.conv = keras.layers.Conv1D(
                filters=self.units,
                kernel_size=1,
                strides=1,
                padding="valid",
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer
            )
        elif len(input_shape) == 4:  # Conv2D, [N,H,W,C]
            self.conv = keras.layers.Conv2D(
                filters=self.units,
                kernel_size=(1,1),
                strides=(1,1),
                padding="valid",
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer
            )
        else:
            raise NotImplementedError(f"Input with shape of {input_shape} not supported, modified this code as needed!")

        # batch normalization
        if self.use_bn:
            self.bn = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=1E-5, center=True, scale=True)

        # build submodules
        super(SharedDense, self).build(input_shape)

    def call(self, inputs, training=False):
        logits = self.conv(inputs, training=training)
        if self.use_bn:
            logits = self.bn(logits, training=training)
        logits = self.activation(logits)
        return logits

    def get_config(self):
        config = {
            "units": self.units,
            "use_bias": self.use_bias,
            "use_bn": self.use_bn,
            "activation": keras.activations.serialize(self.activation),
            "kernel_initializer": self.kernel_initializer
        }
        super_config = super(SharedDense, self).get_config()
        config.update(super_config)
        return config


class AggregateModule(keras.layers.Layer):
    def __init__(self, num_samples, radius, num_neighbours, mlp_units, **kwargs):
        super(AggregateModule, self).__init__(**kwargs)
        self.num_samples = num_samples
        self.radius = radius
        self.num_neighbours = num_neighbours
        self.mlp_units = mlp_units

    def build(self, input_shape):
        # MLP for feature fusion
        mlp_linear_layers = []
        for units in self.mlp_units:
            mlp_linear_layers.append(SharedDense(units, use_bn=True, activation="relu"))
        self.mlp_layer = keras.Sequential(mlp_linear_layers, name="mlp_layer")
        self.pooling_layer = keras.layers.MaxPool2D((1, self.num_neighbours), (1, 1), "VALID", name="max_pool")

        super(AggregateModule, self).build(input_shape)

    def call(self, points, features=None, refs=None, training=False):
        # get reference points (aggregation center)
        if refs is None:
            ref_idx = fps_sampling(points, self.num_samples)  # [B,M]
            ref_points = tf.gather(points, ref_idx, batch_dims=1)  # [B,M,3]
        else:
            assert refs.shape[1] == self.num_samples
            ref_points = refs  # [B,M,3]

        # get neighbours for each reference points
        nn_idx, _ = ball_grouping(points, ref_points, self.num_neighbours, self.radius)  # [B,M,K], _
        nn_points = tf.gather(points, nn_idx, batch_dims=1)  # [B,M,K,3]

        # centralize & normalize nn_points
        nn_points = nn_points - tf.expand_dims(ref_points, 2) # [B,M,K,3], centralize
        nn_points = nn_points / self.radius  # normalize

        # fusion features
        if features is not None:
            nn_features = tf.gather(features, nn_idx, batch_dims=1)  # [B,M,K,C]
            nn_features = tf.concat((nn_points, nn_features), axis=3)  # [B,M,K,3+C]
        else:
            nn_features = nn_points
        nn_features = self.mlp_layer(nn_features, training=training)  # [B,M,K,C']

        # pooling
        ref_features = tf.squeeze(self.pooling_layer(nn_features), axis=2)  # [B,M,C']

        if refs is None:
            return ref_points, ref_features
        else:
            return ref_features


class PropagationModule(keras.layers.Layer):
    def __init__(self, mlp_units, **kwargs):
        super(PropagationModule, self).__init__(**kwargs)
        self.mlp_units = mlp_units

    def build(self, input_shape):
        mlp_linear_layers = []
        for units in self.mlp_units:
            mlp_linear_layers.append(SharedDense(units, use_bn=True, activation="relu"))
        self.mlp_layer = keras.Sequential(mlp_linear_layers)

        super(PropagationModule, self).build(input_shape)

    def call(self, known_points, known_features, unknown_points, unknown_features=None, training=False):
        interpolated_features = interpolate(unknown_points, known_points, known_features, k=3)  # [B,M,C]

        if unknown_features is not None:
            interpolated_features = tf.concat((unknown_features, interpolated_features), axis=2)  # [B,M,C']

        new_features = self.mlp_layer(interpolated_features, training=training)  # [B,M,C]

        return new_features


class BackboneModule(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BackboneModule, self).__init__(**kwargs)
        self.n_list = [2048, 1024, 512, 256]
        self.k_list = [64, 32, 16, 16]
        self.r_list = [0.2, 0.4, 0.8, 1.2]

    def build(self, input_shape):
        self.ds1 = AggregateModule(self.n_list[0], self.r_list[0], self.k_list[0], [64, 64, 128], name="ds_1")
        self.ds2 = AggregateModule(self.n_list[1], self.r_list[1], self.k_list[1], [128, 128, 256], name="ds_2")
        self.ds3 = AggregateModule(self.n_list[2], self.r_list[2], self.k_list[2], [128, 128, 256], name="ds_3")
        self.ds4 = AggregateModule(self.n_list[3], self.r_list[3], self.k_list[3], [128, 128, 256], name="ds_4")

        self.us1 = PropagationModule([256, 128, 256], name="us_1")
        self.us2 = PropagationModule([256, 128, 256], name="us_2")

        super(BackboneModule, self).build(input_shape)

    def call(self, points, features=None, training=False):
        # orginal -> scale1
        scale1_idx = fps_sampling(points, self.n_list[0])
        scale1_pts = tf.gather(points, scale1_idx, batch_dims=1)
        scale1_fts = self.ds1(points, features, scale1_pts, training=training)

        # scale1 -> scale2
        scale2_idx = scale1_idx[:, :self.n_list[1]]
        scale2_pts = scale1_pts[:, :self.n_list[1], :]
        scale2_fts = self.ds2(scale1_pts, scale1_fts, scale2_pts, training=training)

        # scale2 -> scale3
        # scale3_idx = scale1_idx[:, :self.n_list[2]]
        scale3_pts = scale1_pts[:, :self.n_list[2], :]
        scale3_fts = self.ds3(scale2_pts, scale2_fts, scale3_pts, training=training)

        # scale3 -> scale4
        # scale4_idx = scale1_idx[:, :self.n_list[3]]
        scale4_pts = scale1_pts[:, :self.n_list[3], :]
        scale4_fts = self.ds4(scale3_pts, scale3_fts, scale4_pts, training=training)

        # scale4 -> scale3(fused) -> scale2(fused)
        scale3_fts = self.us1(scale4_pts, scale4_fts, scale3_pts, scale3_fts, training=training)
        scale2_fts = self.us2(scale3_pts, scale3_fts, scale2_pts, scale2_fts, training=training)

        return scale2_idx, scale2_pts, scale2_fts


class VoteModule(keras.layers.Layer):
    def __init__(self, input_dims, **kwargs):
        super(VoteModule, self).__init__(**kwargs)
        self.input_dims = input_dims

    def build(self, input_shape):
        # MLP for params generate
        linear_layers = []
        for _ in range(3):
            linear_layers.append(SharedDense(self.input_dims, use_bn=True, activation="relu"))
        linear_layers.append(SharedDense(3 + self.input_dims, use_bias=True, activation=None))
        self.mlp = keras.Sequential(linear_layers)

        super(VoteModule, self).build(input_shape)

    def call(self, points, features, training=False):
        net = self.mlp(features, training=training)  # [B,M,3+C]
        vote_xyz = points + net[:,:,:3]  # [B,M,3]
        vote_features = features + net[:,:,3:]  # [B,M,C]
        return vote_xyz, vote_features


class ProposalModule(keras.layers.Layer):
    def __init__(self, num_class, num_proposal, num_heading_bin, mean_size, **kwargs):
        super(ProposalModule, self).__init__(**kwargs)
        self.num_class = num_class
        self.num_proposal = num_proposal
        self.num_heading_bin = num_heading_bin
        self.mean_size = tf.reshape(mean_size, (1, 1, self.num_class, 3))

    def build(self, input_shape):
        # grouping module
        self.agg_module = AggregateModule(self.num_proposal, 0.3, 16, [128, 128, 128])

        # proposal layer
        mlp_linear_layers = []
        for units in [128, 128, 128]:
            mlp_linear_layers.append(SharedDense(units, use_bn=True, activation="relu"))
        mlp_linear_layers.append(SharedDense(
            units=
                2 +                     # objectness
                3 +                     # object center
                self.num_heading_bin +  # heading bin
                self.num_heading_bin +  # residual for each heading bin
                self.num_class +        # size scores
                self.num_class*3 +      # size residuals normalized
                self.num_class,         # class score
            use_bias=True,
            activation=None,
        ))
        self.proposal_layer = keras.Sequential(mlp_linear_layers)

        super(ProposalModule, self).build(input_shape)

    def call(self, vote_xyz, vote_features, training=False):
        # STEP-1: aggregate & proposal
        agg_points, agg_features = self.agg_module(vote_xyz, vote_features, training=training)
        logits = self.proposal_layer(agg_features, training=training)

        # STEP-2: decode proposal logits
        fence = 0
        # objectness
        objectness = logits[:,:,fence:fence+2]  # [B,P,2]
        fence += 2
        # center
        base_center = agg_points  # [B,P,3]
        center = base_center + logits[:,:,fence:fence+3]  # [B,P,3]
        fence += 3
        # heading
        heading_bin_prob = logits[:,:,fence:fence+self.num_heading_bin]  # [B,P,heading_bins]
        fence += self.num_heading_bin
        heading_residual_normalized = logits[:,:,fence:fence+self.num_heading_bin]  # [B,P,heading_bins]
        fence += self.num_heading_bin
        heading_residual = heading_residual_normalized * (np.pi/self.num_heading_bin)  # [B,P,heading_bins]
        # size
        size_score = logits[:,:,fence:fence+self.num_class]  # [B,P,num_class]
        fence += self.num_class
        size_residual_normalized = tf.reshape(logits[:,:,fence:fence+3*self.num_class], (-1, self.num_proposal, self.num_class, 3))  # [B,P,num_class, 3]
        fence += self.num_class * 3
        size_residual = size_residual_normalized * self.mean_size  # [B,P,num_class, 3]
        # class score
        scores = logits[:,:,fence:fence+self.num_class]  # [B,P,num_class]
        fence += self.num_class

        return \
            objectness, \
            base_center, center, \
            heading_bin_prob, heading_residual_normalized, heading_residual, \
            size_score, size_residual_normalized, size_residual, \
            scores


class DetectModel(keras.Model):
    def __init__(self, num_class, num_proposal, num_heading_bin, mean_size, **kwargs):
        super(DetectModel, self).__init__(**kwargs)
        self.num_class = num_class
        self.num_proposal = num_proposal
        self.num_heading_bin = num_heading_bin
        self.mean_size = mean_size

    def build(self, input_shape):
        self.backbone = BackboneModule(name="backbone")
        self.voter = VoteModule(256, name="voter")
        self.proposer = ProposalModule(self.num_class, self.num_proposal, self.num_heading_bin, self.mean_size, name="proposer")

        super(DetectModel, self).build(input_shape)

    def call(self, inputs, training=False):
        # unpack inputs
        points = inputs[:,:,:3]
        features = inputs[:,:,3:] if inputs.shape[2] > 3 else None

        # backbone
        scene_idx, scene_pts, scene_fts = self.backbone(points, features, training=training)

        # voting
        vote_xyz, vote_features = self.voter(scene_pts, scene_fts, training=training)
        vote_features = vote_features / tf.norm(vote_features, ord=2, axis=2, keepdims=True)

        # proposaling
        objectness, \
        base_center, center, \
        heading_bin_prob, heading_residual_normalized, heading_residual, \
        size_score, size_residual_normalized, size_residual, \
        scores = self.proposer(vote_xyz, vote_features, training=training)

        # return pred_dict
        params = {
            "scene_idx": scene_idx,
            "scene_pts": scene_pts,
            "vote_xyz": vote_xyz,
            "objectness": objectness,
            "base_center": base_center,
            "center": center,
            "heading_bin_prob": heading_bin_prob,
            "heading_residual_normalized": heading_residual_normalized,
            "heading_residual": heading_residual,
            "size_score": size_score,
            "size_residual_normalized": size_residual_normalized,
            "size_residual": size_residual,
            "scores": scores
        }
        return params


if __name__ == "__main__":
    import numpy as np
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)

    def get_mx(*shape):
        # return tf.random.uniform(shape, -30, 30, dtype=tf.float32)
        return tf.convert_to_tensor(np.random.uniform(-30, 30, shape).astype(np.float32))

    pts = get_mx(3,20000,3)
    mean_size = get_mx(18, 3)

    model = DetectModel(num_class=18,
                        num_proposal=128,
                        num_heading_bin=18,
                        mean_size=mean_size)
    res1 = model(pts, training=True)
    res2 = model(pts, training=False)


