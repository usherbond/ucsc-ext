library(lubridate)
library(weatherData)

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

testDate <- function(testDay) {
  myDate <- as.Date(testDay)
  cachedDF <- getPWSData("IYUCATNT2",as.Date(myDate))
  webDF <- getPWSData("IYUCATNT2",as.Date(myDate),T)
  print("Raw compare")
  print(all.equal(cachedDF,webDF))
  print("Equal row compare")
  print(all.equal(webDF[(webDF$Time.1 %in% cachedDF$Time.1),],cachedDF))
}

#test <- getDetailedWeather("IYUCATNT2", "2017-21-01",station_type='id',opt_all_columns=T)
#cachedDF <- getPWSData("IYUCATNT2",as.Date("2017-02-28"))
#webDF <- getPWSData("IYUCATNT2",as.Date("2017-02-28"),T)
#print("Raw compare")
#print(all.equal(cachedDF,webDF))
#print("Equal row compare")
#print(all.equal(webDF[(webDF$Time.1 %in% cachedDF$Time.1),],cachedDF))

testDate("2017-04-01")

#quantile(webDF$WindSpeedMPH,0.95)



