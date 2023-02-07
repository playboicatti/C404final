import re
import random

class MarkovLinked:
    __dic = dict()
    __keyStart = "#start#"
    __keyEnd = "#end#"
    __maxLoop = 500
    __topicMaxLenth = 1000  # 句子长度

    def __init__(self):
        self.__dic[self.__keyStart] = []

    def printMarkovLinked(self):
        for key in self.__dic.keys():
            print('%s\t%s' % (key, self.__dic[key]))

    def append(self, content):
        # clear
        content = re.sub('\s|\n|\t', '', content)
        ie = self.getIterator(content)
        i = 0
        for x in ie:
            key = '%s%s' % (x[0], x[1])
            val = x[2]
            if key not in self.__dic.keys():
                self.__dic[key] = []
            self.__dic[key].append(val)
            # 记录开始 key
            if i == 0:
                self.__dic[self.__keyStart].append(key)
            i += 1

        pass

    def getIterator(self, txt):
        ct = len(txt)
        if ct < 3:
            return
        for i in range(ct - 2 + 1):
            w1 = txt[i]
            w2 = txt[i + 1]
            w3 = txt[i + 2] if i + 2 < ct else self.__keyEnd
            yield (w1, w2, w3)

    def topicBuilder(self, topicMax=0):
        # 随机选择一个开始词
        startKeyArr = self.__dic[self.__keyStart]
        j = random.randint(0, len(startKeyArr) - 1)
        key = startKeyArr[j]  # tuple 类型
        # 待返回的句子
        topic = key

        i = 0

        if topicMax <= 0:
            topicMax = self.__topicMaxLenth

        while i < self.__maxLoop:
            i += 1
            if key not in self.__dic.keys():
                break
            arr = self.__dic[key]
            if not arr:
                break
            j = random.randint(0, len(arr) - 1)
            if j < 0:
                j = 0
            sufix = arr[j]
            # 后缀为结束符时，终止生成
            if sufix == self.__keyEnd:
                break
            # 构建 topic
            topic += sufix
            tLen = len(topic)
            if tLen >= topicMax:
                break
            nextKey = '%s%s' % (key[1], sufix)
            # markovLinked.append(nextKey)
            # 无
            if nextKey not in self.__dic.keys():
                break
            key = nextKey
        # print('markovLinked ',markovLinked)
        return topic


def fileReader():
    path = "markov_data.txt"
    with open(path, 'r', encoding='utf-8') as f:
        rows = 0
        # 按行统计
        while True:
            rows += 1
            line = f.readline()
            if not line:
                print('读取结束 %s' % path)
                return
            print('content rows=%s len=%s type=%s' % (rows, len(line), type(line)))
            yield line
    pass


def main():
    markov = MarkovLinked()
    reader = fileReader()
    for row in reader:
        print(row)
        markov.append(row)
    # markov.printMarkovLinked()
    # 生成句子
    for i in range(200):
        topic = markov.topicBuilder(topicMax=300)
        print('%s\t%s' % (i, topic))
    pass


# main()
if __name__ == '__main__':
    main()
    pass
