class Translation():
    def __init__(self):
        self.REV_CLASS_MAP = {
            0: '1',
            1: '2',
            2: '3',
            3: '4',
            4: '5',
            5: '6',
            6: '7',
            7: '8',
            8: '9',
            9: '10',
        }

    def mapper(self, key):
        return self.REV_CLASS_MAP[key]
