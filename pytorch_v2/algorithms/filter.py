class Filter:
    def __init__(self):
        self.info_lis = []

    def judge(self, string) -> bool:
        for s in self.info_lis:
            if s in string:
                return False
            else:
                continue
        return True
