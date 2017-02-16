package com.community.datascience

import org.apache.spark.sql.SparkSession
import edu.stanford.nlp.ling.CoreAnnotations.{LemmaAnnotation, SentencesAnnotation, TokensAnnotation}
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}
import org.apache.spark.ml.feature.{CountVectorizer, IDF}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import java.util.Properties

case class TextDocument (val id : String, val title : String, val text : String)

class AssembleDocumentTermMatrix (private val spark : SparkSession) extends Serializable {
    import spark.implicits._

    def createNLPPipeline () : StanfordCoreNLP = {
        val props = new Properties()
        props.put("annotators", "tokenize, ssplit, pos, lemma")
        new StanfordCoreNLP(props)
    }

    def isLetterOrDigit (str: String) : Boolean = {
        str.forall(c => Character.isLetterOrDigit(c))
    }

    def titleAndContentToTerms (docs : Dataset[TextDocument], stopWordsFile : String)
        : Dataset[(String, Seq[String])] = {
        val bStopWords = spark.sparkContext.broadcast(spark.read.textFile(stopWordsFile).collect())

        docs.mapPartitions { iter =>
            val pipeline = createNLPPipeline()
            val stopWords = bStopWords.value.toSet
            iter.map { case d : TextDocument => (d.id, 
                plainTextToLemmas(d.title, stopWords, pipeline) ++ plainTextToLemmas(d.text, stopWords, pipeline)) }
        }
    }

    def plainTextToLemmas (text: String, stopWords: Set[String], pipeline : StanfordCoreNLP)
        : Seq[String] = {
        val doc = new Annotation(text)
        pipeline.annotate(doc)
        val lemmas = new ArrayBuffer[String]()
        val sentences = doc.get(classOf[SentencesAnnotation])
        for (sentence <- sentences.asScala;
             token <- sentence.get(classOf[TokensAnnotation]).asScala) {
            val lemma = token.get(classOf[LemmaAnnotation])
            if(lemma.length > 2 && !stopWords.contains(lemma) && isLetterOrDigit(lemma))
                lemmas += lemma.toLowerCase
        }
        lemmas
    }

    def fitAndSaveVocabModel (
        docs : Dataset[TextDocument], 
        stopWordsFile : String, 
        numTerms : Int,
        saveAsFile : String) {
        val terms = titleAndContentToTerms(docs, stopWordsFile)     
        val termDF = terms.toDF("id", "terms")
        val filtered = termDF.where(size($"terms") > 1)
        val countVectorizer = new CountVectorizer().
            setInputCol("terms").
            setOutputCol("termFreqs").
            setVocabSize(numTerms)
    
        val vocabModel = countVectorizer.fit(filtered)

        vocabModel.save(saveAsFile)
    }

    def  documentTermMatrix(docs : Dataset[TextDocument], stopWordsFile : String, numTerms : Int)
        : (DataFrame, Array[String], Map[Long, String], Array[Double]) = {
        val docsWithTerms = titleAndContentToTerms(docs, stopWordsFile)
        val termDF = docsWithTerms.toDF("id", "terms")
        val filtered = termDF.where(size($"terms") > 1)
        val countVectorizer = new CountVectorizer().
            setInputCol("terms").
            setOutputCol("termFreqs").
            setVocabSize(numTerms)
    
        val vocabModel = countVectorizer.fit(filtered)
        val vocabTremFreqs = vocabModel.transform(filtered)
        val docTermFreqs = vocabModel.transform(filtered)
        val terms = vocabModel.vocabulary

        docTermFreqs.cache()

        val docMap = docTermFreqs.select("id").rdd
            .zipWithIndex.map{ case (row, idx) => (idx, row.getAs[String]("id")) }.collect.toMap

        val idf = new IDF().setInputCol("termFreqs").setOutputCol("tfidfVec")
        val idfModel = idf.fit(docTermFreqs)
        val docTermMatrix = idfModel.transform(docTermFreqs).select("id", "tfidfVec")

        (docTermMatrix, terms, docMap, idfModel.idf.toArray)
    }
}