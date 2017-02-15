name := "Community ML Project"

version := "1.0"

scalaVersion := "2.11.8"

resolvers += Resolver.mavenLocal

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.1.0"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.1.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.1.0"
libraryDependencies += "edu.stanford.nlp" % "stanford-corenlp" % "3.7.0"
libraryDependencies += "org.mongodb.spark" %% "mongo-spark-connector" % "2.0.0"