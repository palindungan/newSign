class Translation():
    def __init__(self):
        self.NUMERIC_REV_CLASS_MAP = {
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
        self.ALPHABET_REV_CLASS_MAP = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D',
            4: 'E',
            5: 'F',
            6: 'G',
            7: 'H',
            8: 'I',
            9: 'J',
            10: 'K',
            11: 'L',
            12: 'M',
            13: 'N',
            14: 'O',
            15: 'P',
            16: 'Q',
            17: 'R',
            18: 'S',
            19: 'T',
            20: 'U',
            21: 'V',
            22: 'W',
            23: 'X',
            24: 'Y',
            25: 'Z',
        }
        self.ALL_REV_CLASS_MAP = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D',
            4: 'E',
            5: 'F',
            6: 'G',
            7: 'H',
            8: 'I',
            9: 'J',
            10: 'K',
            11: 'L',
            12: 'M',
            13: 'N',
            14: 'O',
            15: 'P',
            16: 'Q',
            17: 'R',
            18: 'S',
            19: 'T',
            20: 'U',
            21: 'V',
            22: 'W',
            23: 'X',
            24: 'Y',
            25: 'Z',
            26: '1',
            27: '2',
            28: '3',
            29: '4',
            30: '5',
            31: '6',
            32: '7',
            33: '8',
            34: '9',
            35: '10',
        }

    def mapper(self, key, predictionType):
        if predictionType == 'NUMERIC':
            return self.NUMERIC_REV_CLASS_MAP[key]
        elif predictionType == 'ALPHABET':
            return self.ALPHABET_REV_CLASS_MAP[key]
        elif predictionType == 'ALL':
            return self.ALL_REV_CLASS_MAP[key]
        else:
            return None
