package com.group12.demo

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

object JavaNaiveBayesExample {
  case class Car (buying : String, maint : String, doors : String, persons : String, lug_boot : String, safety : String, carClass : String)
  def main(args:Array[String]){
    // Path of hadoop home directory
//    System.setProperty("hadoop.home.dir", "C:\\Hadoop\\hadoop-common-2.2.0-bin-master\\")
    
    //Parsing the input data
    def parseData(str:String) : Car = {
      var line = str.split(",")
      
      if(line.length == 7)
        Car(line(0), line(1), line(2), line(3), line(4), line(5), line(6))
      else Car("None", "None", "None", "None", "None", "None", "None")
    }
    
    
    //Setting up Spark configurations
    val conf = new SparkConf().setAppName("SparkAction").setMaster("local")
    val sc = new SparkContext(conf)
    
    //Reading an input data file
    val inputDataRDD = sc.textFile(args(0))
    
    val parsedInputRDD = inputDataRDD.map(parseData).cache()
    val validParsedInputRDD = parsedInputRDD.filter(line => !line.carClass.equals("None"))
    
    //Converting Strings to Double
    var buyingMap : Map[String,Double] = Map()
    var index1 = 0.0
    validParsedInputRDD.map(car => car.buying).distinct.collect().foreach(x => { buyingMap += (x -> index1); index1 += 1.0 })
    
    var maintMap : Map[String,Double] = Map()
    var index2 = 0.0
    validParsedInputRDD.map(car => car.maint).distinct.collect().foreach(x => { maintMap += (x -> index2); index2 += 1.0 })
    
    var doorsMap : Map[String,Double] = Map()
    var index3 = 0.0
    validParsedInputRDD.map(car => car.doors).distinct.collect().foreach(x => { doorsMap += (x -> index3); index3 += 1.0 })
    
    var personsMap : Map[String,Double] = Map()
    var index4 = 0.0
    validParsedInputRDD.map(car => car.persons).distinct.collect().foreach(x => { personsMap += (x -> index4); index4 += 1.0 })
    
    var lugMap : Map[String,Double] = Map()
    var index5 = 0.0
    validParsedInputRDD.map(car => car.lug_boot).distinct.collect().foreach(x => { lugMap += (x -> index5); index5 += 1.0 })
    
    var safetyMap : Map[String,Double] = Map()
    var index6 = 0.0
    validParsedInputRDD.map(car => car.safety).distinct.collect().foreach(x => { safetyMap += (x -> index6); index6 += 1.0 })
    
    var classMap : Map[String,Double] = Map()
    var index7 = 0.0
    validParsedInputRDD.map(car => car.carClass).distinct.collect().foreach(x => { classMap += (x -> index7); index7 += 1.0 })

    
    //Getting final data for Decision tree
    val dataPrep = validParsedInputRDD.map(car => {
      val carClass = classMap(car.carClass)
      val buying = buyingMap(car.buying)
      val maint = maintMap(car.maint)
      val doors = doorsMap(car.doors)
      val persons = personsMap(car.persons)
      val lugBoot = lugMap(car.lug_boot)
      val safety = safetyMap(car.safety)
      
      Array(carClass.toDouble,buying.toDouble,maint.toDouble,doors.toDouble,persons.toDouble,lugBoot.toDouble,safety.toDouble)
    })
    
   
    //Creating vector from the data
    val dataLabels = dataPrep.map{dataLine => 
      LabeledPoint(dataLine.apply(0).toDouble,Vectors.dense(dataLine.apply(1), dataLine.apply(2), dataLine.apply(3), dataLine.apply(4), dataLine.apply(5), dataLine.apply(6))  
    )}
    
    
    // Split the data into training and test sets (30% held out for testing)
    val splits = dataLabels.randomSplit(Array(0.7, 0.3),seed = 11L)
    val trainingData = splits(0)
    val testData = splits(1)
    

val model = NaiveBayes.train(trainingData, lambda = 1.0, modelType = "multinomial")

val predictionAndLabel = testData.map(p => (model.predict(p.features), p.label))
val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / testData.count()

predictionAndLabel.collect().foreach(x => println(x._1 + " - " + x._2))   
predictionAndLabel.saveAsTextFile(args(1))
  }
    
}