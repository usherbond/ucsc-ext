library(lubridate)
library(weatherData)
library(zoo)

getPWSData <- function(stationName,dateQuerry,forceWebQuerry=FALSE) {
  if (!is.character(stationName)) {
    stop("stationName should be a string")
  }
  if (!is.Date(dateQuerry)) {
    stop("dateQuery should be a Date")
  }
  # Extract weather info:
  yearStr <- format(dateQuerry,format="%Y")
  monthStr <- format(dateQuerry,format="%m")
  #print(yearStr)
  #print(monthStr)
  # Assuming cache is in the current execution path otherwise we could use ~
  dirStr <- sprintf("weatherc/%s/%s/%s",stationName,yearStr,monthStr)
  # Filename still uses the whole date if we want to flat the dir struct
  fileName <- sprintf("%s-%s.csv",stationName,as.character(dateQuerry))
  #print(dirStr)
  #print(fileName)
  filePath <- paste(dirStr,fileName,sep="/")
  print(filePath)
  if( file.exists(filePath) && !forceWebQuerry ) {
    print("File exists!")
    cachedDF <- read.csv(filePath,stringsAsFactors=FALSE)
    cachedDF <- transform(cachedDF,Time=as.POSIXct(Time))
    #print(head(cachedDF))
    return(cachedDF)
  } else {
    print("File missing!")
    if(!dir.exists(dirStr)) {
      dir.create(dirStr,recursive=TRUE)
    }
    #weatherDF <- data.frame(a=c(1,2,3),b=c(4,5,6))
    weatherDF <- getDetailedWeather(stationName, dateQuerry,station_type='id',opt_all_columns=TRUE)
    Sys.sleep(1)
    #print(head(weatherDF))
    # Caching code only executed if the force flag is not set
    if (!forceWebQuerry) {
      #write.csv(weatherDF,filePath)
      print(sprintf("Writing %d to file %s",nrow(weatherDF),filePath))
      write.csv(weatherDF,filePath,row.names=FALSE)
    }
    return(weatherDF)
  }
}

convertPWSData2Zoo <- function(pwsDF) {
  if (!is.data.frame(pwsDF)) {
    stop("input pwsDF should be a data.frame")
  }
  timeKeeperColName <- "Time"
  timeIdx <- which(colnames(pwsDF)==timeKeeperColName)
  # Names of columns to exclude includding the time keeping one
  excludeCols <- c(timeKeeperColName,"Time.1")
  #excludeIdx <- which(colnames(pwsDF) %in% excludeCols)
  #Zoo only seems to work with numbers so removing any character from DF:
  #excludeCols <- !((colnames(pwsDF) %in% excludeCols)|sapply(pwsDF,is.character))
  excludeCols <- !(colnames(pwsDF) %in% excludeCols)&(sapply(pwsDF,is.numeric))
  #print(timeIdx)
  #print(excludeIdx)
  myZoo <- zoo(pwsDF[,excludeCols],pwsDF[,timeIdx])
  if (mode(coredata(myZoo)) != "numeric") {
    stop("Final zoo matrix is expected to have numeric mode");
  }
  #if (mode(time(myZoo)) != "numeric") {
  #  stop("Final zoo time is expected to have numeric mode");
  #}
  importantCols <- c("WindDirectionDegrees","WindSpeedMPH", "WindSpeedGustMPH")
  if (!all(importantCols %in% colnames(myZoo))) {
    stop("Not all the important fields are present in the Zoo")
  }
#print(res)
  return(myZoo)
}

getPWSZoo <- function(...) {
  return(convertPWSData2Zoo(getPWSData(...)))
}

testDate <- function(testDay) {
  myDate <- as.Date(testDay)
  cachedDF <- getPWSData("IYUCATNT2",as.Date(myDate))
  webDF <- getPWSData("IYUCATNT2",as.Date(myDate),T)
  print("Raw compare")
  print(all.equal(cachedDF,webDF))
  print("Equal row compare")
  print(all.equal(webDF[(webDF$Time.1 %in% cachedDF$Time.1),],cachedDF))
}

retList <- function() {
  return(list(1,2))
}

mySplitDate <- function(dateQuerry) {
  if (!is.Date(dateQuerry)) {
    stop("dateQuery should be a Date")
  }
  return(list(year=year(dateQuerry),month=month(dateQuerry),day=day(dateQuerry)))
}

processDate <- function(dateQuerry) {
  if (!is.Date(dateQuerry)) {
    stop("dateQuery should be a Date")
  }
  # The following function can't be called in parallel:
  weatherDF <- getPWSData("IYUCATNT2",dateQuerry)
  # Processing the DF to obtain different statistics:
  results <- list(
    speedQ95 = quantile(weatherDF$WindSpeedMPH,0.95),
    maxSpeed = max(weatherDF$WindSpeedMPH),
    avgDir = mean(weatherDF$WindDirectionDegrees),
    # Following is inacurate but it is just to get a string:
    medDir = median(weatherDF$WindDirection,na.rm=TRUE)
  )
  #return(results$speedQ95)
  return(results)
}

processDates <- function(dateQuerry) {
  if (!is.Date(dateQuerry)) {
    stop("dateQuery should be a Date")
  }
  resultMatrix <- sapply(dateQuerry,processDate)
  # Transpose seems to work even when we have mixed types
  return(t(resultMatrix))
}

#test <- getDetailedWeather("IYUCATNT2", "2017-21-01",station_type='id',opt_all_columns=T)
#cachedDF <- getPWSData("IYUCATNT2",as.Date("2017-02-28"))
#webDF <- getPWSData("IYUCATNT2",as.Date("2017-02-28"),T)
#print("Raw compare")
#print(all.equal(cachedDF,webDF))
#print("Equal row compare")
#print(all.equal(webDF[(webDF$Time.1 %in% cachedDF$Time.1),],cachedDF))

#testDate("2017-04-01")

#quantile(webDF$WindSpeedMPH,0.95)

# This is 
#http://climate.umn.edu/snow_fence/components/winddirectionanddegreeswithouttable3.htm

myZoo<- getPWSZoo("IYUCATNT2",as.Date("2014-12-24"))
zooTimes <- time(myZoo)
print(zooTimes[14]-zooTimes[3])
plot(myZoo$WindSpeedMPH)


