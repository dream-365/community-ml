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

class LSAQueryEngine (
    val svd : SingularValueDecomposition[RowMatrix, Matrix], 
    val termIds : Array[String],
    val docIds : Map[Long, String],
    val termIdfs : Array[Double]) {

    val US : RowMatrix = multiplyRowMatrixByDiagnoal(svd.U, svd.s)
    val normalizedUS : RowMatrix = normalizeDistributedRows(US)
    
    def normalizeDistributedRows (mat : RowMatrix) : RowMatrix = {
        new RowMatrix(mat.rows.map { row =>  
            val array = row.toArray
            val length = math.sqrt(array.map { v => v * v }.sum)
            Vectors.dense(array.map(_ / length ).toArray)
        })
    }

    def multiplyMatrixByDiagnoal (mat : Matrix, diag : MLLibVector) : BDenseMatrix[Double] = {
        val mArray = mat.toArray
        val dArray = diag.toArray
        val denseMatrix = new BDenseMatrix(mat.numRows, mat.numCols, mArray)
        denseMatrix.mapPairs { case ((r, c), v) => v * dArray(c) }
    }

    def multiplyRowMatrixByDiagnoal (mat : RowMatrix, diag : MLLibVector) : RowMatrix = {
        val dArray = diag.toArray
        new RowMatrix(mat.rows.map { vec =>
            val vecArray = vec.toArray
            val newArray = (0 until vec.size).map(i => vecArray(i) * dArray(i)).toArray
            Vectors.dense(newArray)
        })
    }

    def topDocsForDocs(docId : Long) : Seq[(Double, Long)] = {
        val docArray = normalizedUS.rows.zipWithIndex.map(_.swap).lookup(docId).head.toArray
        val docMatrix = Matrices.dense(docArray.length, 1, docArray)
        val allDocScores = normalizedUS.multiply(docMatrix)
        val allDocWeights = allDocScores.rows.map(_.toArray(0)).zipWithIndex
        allDocWeights.filter(!_._1.isNaN).top(10)
    }
}