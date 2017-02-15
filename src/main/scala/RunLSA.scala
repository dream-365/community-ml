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

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.mllib.linalg.{SparseVector, DenseVector}

// curl -O https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2
// bzip2 -cd 
// hadoop fs -put - /user/ds/wikidump.xml
// $SPARK_HOME/bin/spark-shell --packages edu.stanford.nlp:stanford-corenlp:3.7.0 --master spark://10.0.0.4:7077 --driver-memory 4G --executor-memory 8G
// http://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/$version/stanford-corenlp-$version-models.jar


/*
./bin/spark-shell \
--master spark://node1.lab.net:7077 --driver-memory 4G --executor-memory 8G \
--packages edu.stanford.nlp:stanford-corenlp:3.7.0,org.mongodb.spark:mongo-spark-connector_2.11:2.0.0 \
--jars /opt/jars/stanford-corenlp-3.7.0-models-english.jar,/home/jec/jars/lsa-project_2.11-1.0.jar
*/



object RunLSA {
    def main (args : Array[String]) : Unit = {  
        val spark = SparkSession.
                        builder().
                        appName("CML Application").
                        enableHiveSupport().
                        getOrCreate()       
        val numTerms = 10000
        val k = 5000
        val stopWordsFileUri = "hdfs://node1.lab.net:9000/data/stopwords.txt"
        val readConfig = ReadConfig(Map("uri"->"mongodb://cas:cas$mongodb@node1.lab.net,node2.lab.net/cas.msdn_technet_questions_uwp?replicaSet=rs0"))
        val questionDF = MongoSpark.load(spark, readConfig)

        import spark.implicits._

        val docs = questionDF.select("title", "text").as[TextDocument]
        val assembleMatrix = new AssembleDocumentTermMatrix(spark)

        import assembleMatrix._

        val (docTermMatrix, termIds, docIds, termIdfs) = documentTermMatrix(docs, stopWordsFileUri, numTerms)

        val vecRdd = docTermMatrix.rdd.map { row =>
            val sparseVec = row.getAs[MLVector]("tfidfVec").toSparse
            Vectors.sparse(sparseVec.size, sparseVec.indices, sparseVec.values)
        }

        vecRdd.cache()

        val mat = new RowMatrix(vecRdd)
        val svd = mat.computeSVD(k, computeU = true)

        val topConceptTerms = topTermsInTopConcept(svd, 10, 10, termIds)
        val topConceptDocs = topDocsInTopConcept(svd, 10, 10, docIds)

        for((terms, docs) <- topConceptTerms.zip(topConceptDocs)) {
            println("Concept Terms: " + terms.map(_._1).mkString(", "))
            println("Concept Docs: " + docs.map(_._1).mkString(", "))
        }

        val engine = new LSAQueryEngine(svd, termIds, docIds, termIdfs)

        engine.topDocsForDocs(1000).map { case (weight, id) => docIds(id) }.foreach(println)
    }

    def printDocTitles (ids : Seq[Long], docIds : Map[Long, String]) = {
        ids.map(docIds(_)).foreach(println)
    }


    def topTermsInTopConcept (svd : SingularValueDecomposition[RowMatrix, Matrix],
        numConcepts : Int,
        numTerms : Int,
        termIds : Array[String]      
    ) : Seq[Seq[(String, Double)]] = {
        val v = svd.V
        val topTerms = new ArrayBuffer[Seq[(String, Double)]]
        val arr = v.toArray

        for (i <- 0 until numConcepts) {
            val offset = i * v.numRows
            val termWeights = arr.slice(offset, offset + v.numRows).zipWithIndex
            val sorted = termWeights.sortBy(-_._1)
            topTerms += sorted.take(numTerms).map { case (score, id) => (termIds(id), score) }
        }

        topTerms
    }

    def topDocsInTopConcept (svd : SingularValueDecomposition[RowMatrix, Matrix], 
        numConcepts : Int,
        numDocs : Int,
        docIds : Map[Long, String]
    ) : Seq[Seq[(String, Double)]] = {
        val u = svd.U
        val topDocs = new ArrayBuffer[Seq[(String, Double)]]
        
        for (i <- 0 until numConcepts) {
            val docWeights = u.rows.map(_.toArray(i)).zipWithIndex
            topDocs += docWeights.top(numDocs).map { case (score, id) => (docIds(id), score) }
        }

        topDocs
    }

    def calculateDistance () = {
        mat.rows.map { vec => 
            val a1 = vec.toArray
            val a2 = docVec
            var value : Double = 0;
            for( i <- 0 until a1.size) {
                value = value + math.abs(a1(i) - a2(i))
            }
            -value
        }
    }
}