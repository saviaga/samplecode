import csv
import pickle
import operator

from sklearn.feature_extraction.text import TfidfVectorizer


def getPercentage(totalSize, topDesired):
    result = topDesired * totalSize
    result = ((result) / (100))
    return result


def readData():
    authors = {}
    callsToActionAuthor = {}
    with open('filteredResults.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)

        for row in csvreader:
            callToActionText = row[2]
            author = row[5]
            callsToActionAuthor.setdefault(author, [])
            callsToActionAuthor[author].append(callToActionText)
            author = row[5]
            authors.setdefault(author, 0)
            authors[author] += 1

    pickle.dump(authors, open("leadersCalltoAction.p", "w"))
    pickle.dump(callsToActionAuthor, open("leadersCalltoActionText.p", "w"))


def getVectorsAuthor(authorData):
    vectorAuthor = []
    numPosts = len(authorData)
    numWordsPosts = []

    for text in authorData:
        numWords = len(text)
        numWordsPosts.append(int(numWords))
    numWordsPosts.sort()
    medianIndex = numPosts / 2
    medianNumWords = numWordsPosts[medianIndex]
    vectorAuthor.append(numPosts)
    vectorAuthor.append(medianNumWords)
    keyAuthor = str(numPosts) + "," + str(medianNumWords)
    return vectorAuthor, keyAuthor


def correctAuthorData():
    callsToActionAuthor = pickle.load(open("leadersCalltoActionText.p", "r"))
    authors = pickle.load(open("leadersCalltoAction.p", "r"))
    authorsC = {}
    callsToActionAuthorCorrected = {}

    for a in callsToActionAuthor:
        authorData = callsToActionAuthor[a]
        for text in authorData:
            if not text == "":
                authorsC.setdefault(a, 0)
                callsToActionAuthorCorrected.setdefault(a, [])
                authorsC[a] += 1
                callsToActionAuthorCorrected[a].append(text)
    pickle.dump(authorsC, open("leadersCalltoActionCorrected.p", "w"))
    pickle.dump(callsToActionAuthorCorrected, open("callsToActionAuthorCorrectedText.p", "w"))


def understandTypesOfCallsToAction():
    callsToActionAuthor = pickle.load(open("callsToActionAuthorCorrectedText.p", "r"))
    authors = pickle.load(open("leadersCalltoActionCorrected.p", "r"))
    vectorsAuthors = {}
    vectorsAuthorsArray = []
    vectorAuthorsDict = {}
    for author in callsToActionAuthor:
        authorData = callsToActionAuthor[author]
        v, keyAuthor = getVectorsAuthor(authorData, authors[author])
        vectorsAuthors[author] = v
        vectorsAuthorsArray.append(v)
        vectorAuthorsDict.setdefault(keyAuthor, [])
        vectorAuthorsDict[keyAuthor].append(author)

    pickle.dump(vectorsAuthorsArray, open("vectorsAuthorsArray.p", "w"))
    pickle.dump(vectorsAuthors, open("vectorsAuthors.p", "w"))
    pickle.dump(vectorAuthorsDict, open("vectorAuthorsDict.p", "w"))
    return vectorsAuthors, vectorsAuthorsArray, vectorAuthorsDict


def getTopCallToActionLeaders(authors):
    print(len(authors))
    sortedAuthors = sorted(authors.items(), key=operator.itemgetter(1), reverse=True)
    i = 0
    v = getPercentage(len(authors), 5)
    for a, value in sortedAuthors:
        if i < v:
            print(a + "," + str(value))
            i += 1
        else:
            break

    print(v)


def getClusterData():
    vectorAuthorsDict = pickle.load(open("vectorAuthorsDict.p", "rb"))
    callsToActionAuthor = pickle.load(open("callsToActionAuthorCorrectedText.p", "rb"))
    textClusters = {}
    textoCorpus = []
    clusters = pickle.load(open("clustersStored.p", "rb"))

    for c in clusters:
        authors = clusters[c]
        textoGrupo = []
        for a in authors:

            if a in vectorAuthorsDict:
                authors = vectorAuthorsDict[a]
                for ac in authors:
                    articles = callsToActionAuthor[ac]
                    for text in articles:
                        text = text.lower()
                        textoGrupo.append(text)
                        textoCorpus.append(text)
        textClusters[c] = textoGrupo

    pickle.dump(textClusters, open("textClusters.p", "wb"))
    pickle.dump(textoCorpus, open("textoCorpus.p", "wb"))


def calculateFrequentWords():
    textoCorpus = pickle.load(open("textoCorpus.p", "rb"))
    textClusters = pickle.load(open("textClusters.p", "rb"))
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    for c in textClusters:
        print(c)
        oraciones = textClusters[c]
        for o in oraciones:
            print(o)
            vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word', stop_words='english', vocabulary=o)
            vect.fit_transform(textoCorpus)
            idf = vect.idf_
            print(dict(zip(vect.get_feature_names(), idf)))
        vocabulary = "a list of words I want to look for in the documents".split()
        vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word', stop_words='english',
                               vocabulary=vocabulary)
        vect.fit(textoCorpus)
        corpus_tf_idf = vect.transform(textoCorpus)
        print(corpus_tf_idf)
        X = vectorizer.fit_transform(corpus)
        Indexf = vectorizer.idf_
        print(dict(zip(vectorizer.get_feature_names(), idf)))
        features_train_transformed = vectorizer.fit_transform(oraciones)
        vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word', stop_words='english',
                               vocabulary=oraciones)
        vect.fit("fre me from")
        print(features_train_transformed)
        features_train_transformed = selector.transform(features_train_transformed).toarray()
        print(features_train_transformed)


def reviewData():
    callsToActionAuthor = pickle.load(open("callsToActionAuthorCorrectedText.p", "rb"))
    authors = pickle.load(open("leadersCalltoActionCorrected.p", "rb"))
    sortedAuthors = sorted(authors.items(), key=operator.itemgetter(1), reverse=True)
    i = 0
    for a, v in sortedAuthors:
        if i < 5:
            print(a + "," + str(v))
        else:
            break
        i += 1


def understandClusters():
    textClusters = pickle.load(open("textClusters.p", "rb"))
    oraciones = textClusters[2]
    FILE = open("ArticlosCluster2.txt", 'w')
    for o in oraciones:
        FILE.write(o + "\n\n")
    FILE.close()


if __name__ == "__main__":
    understandClusters()
    getClusterData()
    calculateFrequentWords()

    reviewData()
    correctAuthorData()
    vectorsAuthors, vectorsAuthorsArray, vectorAuthorsDict = understandTypesOfCallsToAction()
    print(vectorsAuthorsArray)
    print(len(vectorsAuthorsArray))
    readData()
