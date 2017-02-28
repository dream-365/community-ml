package com.community.datascience

import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.rdd.RDD
import edu.stanford.nlp.ling.CoreAnnotations.{LemmaAnnotation, SentencesAnnotation, TokensAnnotation}
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}
import org.apache.spark.mllib.linalg.{Matrices, Matrix, SingularValueDecomposition, Vectors, Vector => MLLibVector}
import org.apache.spark.ml.linalg.{Vector => MLVector}
import breeze.linalg.{DenseMatrix => BDenseMatrix, SparseVector => BSparseVector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import com.mongodb.spark._
import com.mongodb.spark.config._
import java.util.Properties
import scala.collection.JavaConverters._
import org.bson.Document

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.mllib.linalg.{SparseVector, DenseVector}

// curl -O https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2
// bzip2 -cd 
// hadoop fs -put - /user/ds/wikidump.xml
// $SPARK_HOME/bin/spark-shell --packages edu.stanford.nlp:stanford-corenlp:3.7.0 --master spark://10.0.0.4:7077 --driver-memory 4G --executor-memory 8G
// http://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/3.7.0/stanford-corenlp-3.7.0-models-english.jar


/*
$SPARK_HOME/bin/spark-shell \
--master local[4] \
--driver-memory 8G --executor-memory 16G \
--packages edu.stanford.nlp:stanford-corenlp:3.7.0,org.mongodb.spark:mongo-spark-connector_2.11:2.0.0 \
--jars /home/spark/jars/stanford-corenlp-3.7.0-models-english.jar,/home/spark/jars/community-ml-project_2.11-1.0.jar

$SPARK_HOME/bin/spark-submit \
--class com.community.datascience.RunLSA \
--master spark://prod1.ca.net:7077 \
--driver-memory 4G --executor-memory 4G \
--packages edu.stanford.nlp:stanford-corenlp:3.7.0,org.mongodb.spark:mongo-spark-connector_2.11:2.0.0 \
--jars /home/spark/jars/stanford-corenlp-3.7.0-models-english.jar \
/home/ca/community-ml-project_2.11-1.0.jar
*/

object RunLSA {
    def main (args : Array[String]) : Unit = {  
        val spark = SparkSession.
                        builder().
                        appName("CML Application").
                        enableHiveSupport().
                        getOrCreate()       
        val (engine, terms, docMap) = build(spark, 50000, 500)
        val docInverseMap = docMap.map{ kv => (kv._2, kv._1)}.toMap
        val readConfig = ReadConfig(
            Map("uri"->"mongodb://minint-qvps4a4:27017,vmas-svr001:27017/community.lsa_tasks?replicaSet=rs0"))
        val writeConfig = WriteConfig(
            Map("uri"->"mongodb://minint-qvps4a4:27017,vmas-svr001:27017/community.lsa_results?replicaSet=rs0"))
        import spark.implicits._
        val tasks = MongoSpark.load(spark, readConfig).select("id").as[String].collect.toSeq
        tasks.foreach(id => {
             MongoSpark.save(spark.sparkContext.parallelize(engine.topDocsForDocs(docInverseMap(id)).
                                map { case (weight : Double, idx : Long) => 
                                    new Document(Map[String,Object](
                                        "id" -> id, 
                                        "relateId" -> docMap(idx), 
                                        "weight" -> weight.asInstanceOf[AnyRef]).asJava)
                                    }), writeConfig)
        })

        println("complete")
    }
 
    def build (spark : SparkSession, numTerms : Int, k : Int) : (LSAQueryEngine, Array[String], Map[Long, String]) = {
        val stopWordsFileUri = "/home/spark/data/stopwords.txt"
        val readConfig = ReadConfig(
            Map("uri"->"mongodb://minint-qvps4a4:27017,vmas-svr001:27017/community.msdn_technet_questions?replicaSet=rs0&readPreference=secondary"))
        val questionDF = MongoSpark.load(spark, readConfig).repartition(8)

        import spark.implicits._

        val docs = questionDF.select("id", "title", "text").as[TextDocument]
        val assembleMatrix = new AssembleDocumentTermMatrix(spark)

        import assembleMatrix._

        val (docTermMatrix, terms, docMap, termIdfs) = documentTermMatrix(docs, stopWordsFileUri, numTerms)

        val vecRdd = docTermMatrix.rdd.map { row =>
            val sparseVec = row.getAs[MLVector]("tfidfVec").toSparse
            Vectors.sparse(sparseVec.size, sparseVec.indices, sparseVec.values)
        }

        vecRdd.cache()

        val mat = new RowMatrix(vecRdd)
        val svd = mat.computeSVD(k, computeU = true)

        val topConceptTerms = topTermsInTopConcept(svd, 10, 10, terms)
        val topConceptDocs = topDocsInTopConcept(svd, 10, 10, docMap)

        for((ts, docs) <- topConceptTerms.zip(topConceptDocs)) {
            println("Concept Terms: " + ts.map(_._1).mkString(", "))
            println("Concept Docs: " + docs.map(_._1).mkString(", "))
        }

        val engine = new LSAQueryEngine(svd, terms, docMap, termIdfs)

        (engine, terms, docMap)
    }

    def printDocTitles (ids : Seq[Long], docMap : Map[Long, String]) = {
        ids.map(docMap(_)).foreach(println)
    }

    def topTermsInTopConcept (svd : SingularValueDecomposition[RowMatrix, Matrix],
        numConcepts : Int,
        numTerms : Int,
        terms : Array[String]      
    ) : Seq[Seq[(String, Double)]] = {
        val v = svd.V
        val topTerms = new ArrayBuffer[Seq[(String, Double)]]
        val arr = v.toArray

        for (i <- 0 until numConcepts) {
            val offset = i * v.numRows
            val termWeights = arr.slice(offset, offset + v.numRows).zipWithIndex
            val sorted = termWeights.sortBy(-_._1)
            topTerms += sorted.take(numTerms).map { case (score, id) => (terms(id), score) }
        }

        topTerms
    }

    def topDocsInTopConcept (svd : SingularValueDecomposition[RowMatrix, Matrix], 
        numConcepts : Int,
        numDocs : Int,
        docMap : Map[Long, String]
    ) : Seq[Seq[(String, Double)]] = {
        val u = svd.U
        val topDocs = new ArrayBuffer[Seq[(String, Double)]]  
        for (i <- 0 until numConcepts) {
            val docWeights = u.rows.map(_.toArray(i)).zipWithIndex
            topDocs += docWeights.top(numDocs).map { case (score, id) => (docMap(id), score) }
        }
        topDocs
    }
}