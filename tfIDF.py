
import pickle
import operator

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer


# considering 3 clusters
def getInclusiveText(oracionesA, oracionesB):
    textoCorpus = []
    for o in oracionesA:
        textoCorpus.append(o)
        break
    for o in oracionesB:
        textoCorpus.append(o)
        break
    return textoCorpus


def getCorpusData():
    uniqueWords = {}
    uniqueWordsClusters = {}
    textClusters = pickle.load(open("textClusters.p", "rb"))

    for c in textClusters:
        uniqueWordsClusters.setdefault(c, [])
        oraciones = textClusters[c]

        for o in oraciones:
            words = o.split()
            uniqueWordsSentence = {}
            for w in words:
                uniqueWords.setdefault(w, 0)
                uniqueWords[w] += 1
                uniqueWordsSentence[w] = 0
            uniqueSentence = ""
            for w in uniqueWordsSentence:
                uniqueSentence += w
                uniqueSentence += " "
            uniqueWordsClusters[c].append(w)

    arrayCorpus = []
    for w in uniqueWords:
        arrayCorpus.append(w)
    print(len(arrayCorpus))
    print(len(uniqueWords))
    return arrayCorpus, uniqueWordsClusters


def getTFIDF():
    uniqueWords, uniqueWordsClusters = getCorpusData()

    # vocabulary = "a maga list of words I want to look for in the documents".split()
    # vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word',stop_words='english', vocabulary=vocabulary)
    textoCorpus = pickle.load(open("textoCorpus.p", "rb"))
    # X = vect.fit_transform(textoCorpus)
    # idf =vect.idf_
    # print dict(zip(vect.get_feature_names(), idf))

    tfIDFClusters = {}
    textClusters = pickle.load(open("textClusters.p", "rb"))
    for c in uniqueWordsClusters:
        tfIDFClusters.setdefault(c, [])
        print("clusteer:" + str(c))
        oraciones = uniqueWordsClusters[c]
        for o in oraciones:
            # print "ORACION:"+str(o)
            o = o.split()

            vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word', stop_words='english', vocabulary=o)
            X = vect.fit_transform(textoCorpus)

            idf = vect.idf_
            # print dict(zip(vect.get_feature_names(), idf))
            tfIDFClusters[c].append(dict(zip(vect.get_feature_names(), idf)))
        # break
    # break
    pickle.dump(tfIDFClusters, open("tfIDFClusters.p", "wb"))


def testTFIDFClusters():
    tfIDFClusters = pickle.load(open("tfIDFClusters.p", "rb"))
    wordsTFIDFClusters = {}
    stopWords = set(stopwords.words('english'))
    # {'you': 7.5750758405996201}, {'you': 7.5750758405996201}
    for c in tfIDFClusters:
        print("Cluster:" + str(c))
        wordsTFIDF = {}
        arrayTFIDF = tfIDFClusters[c]
        for dictOracion in arrayTFIDF:
            for w in dictOracion:
                wordsTFIDF[w] = dictOracion[w]
        print("Vocabulary size:" + str(len(wordsTFIDF)))
        wordsTFIDFClusters[c] = wordsTFIDF
        sortedTFIDF = sorted(wordsTFIDF.items(), key=operator.itemgetter(1), reverse=True)
        i = 0
        stopWords.add("about")
        stopWords.add("about.")
        stopWords.add("yet")
        stopWords.add("that")
        stopWords.add("that.")
        stopWords.add("8")
        stopWords.add("2")
        stopWords.add("every")
        stopWords.add("ft.")
        stopWords.add("came")
        stopWords.add("otherwise")
        stopWords.add("...yes")
        stopWords.add("whos")
        stopWords.add("goin")
        stopWords.add("bunch")
        stopWords.add("genuinely")
        stopWords.add("whole")
        stopWords.add("first")
        stopWords.add("nothing.and")
        stopWords.add("many")
        stopWords.add("elsewhere")

        #
        for w, v in sortedTFIDF:
            if not w in stopWords:
                if i < 30:
                    print(w + " ," + str(v))
                    i += 1
                else:
                    break

if __name__=="__main__":

    testTFIDFClusters()
    getCorpusData()
    getTFIDF()


