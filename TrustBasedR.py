class TrustBasedR:
    def __init__(self):
        self.read_data()

    def read_data(self, fn = 'trust_data.txt'):
        with open(fn,'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

            self.data = list()
            for line in lines:
                nums = line.strip().split()
                self.data.append((nums[0], nums[1], nums[2]))

            f.close()