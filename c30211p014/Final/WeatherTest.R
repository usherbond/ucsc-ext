library(lubridate)
#library(weatherData)
library(zoo)
library(dplyr)
library(openair)
library(maptools)
library(ggplot2)
library(ggmap)

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
    Sys.sleep(2)
    #print(head(weatherDF))
    # Caching code only executed if the force flag is not set
    if (!forceWebQuerry && !is.null(weatherDF)) {
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

#Dummy
#getDetailedWeather <- function(...) {return(NULL)}

# This function is just intended to be used for the raw data analysis
getRawData <- function(stationName, startDate, endDate=startDate) {

  dates <- seq(as.Date(startDate), as.Date(endDate), by="days")

  dirStr <- sprintf("weatherc/%s",stationName)
  fileName <- sprintf("%s_bad_dates.txt",stationName)
  filePath <- paste(dirStr,fileName,sep="/")
  #print(filePath)
  missingNum <- 0
  # 87 records total
  if( file.exists(filePath) ) {
    badDates <- as.Date(readLines(filePath))
    goodIdx <- !(dates %in% badDates)
    missingNum <- missingNum + sum(!goodIdx)
    dates <- dates[goodIdx]
  }

  #cheating:
  rec <- lapply(dates,function(x) {
    tmpdf <- getPWSData(stationName,x)
    if (is.null(tmpdf)) {return(NULL)}
    testData <- select(tmpdf, Time, WindDirectionDegrees, WindSpeedMPH, WindSpeedGustMPH)
    return(testData)
  })
  nullRecords <- sum(sapply(rec,is.null))
  missingNum <- missingNum + nullRecords
  #print(sprintf("The number of missing day records from %s to %s in %s is %d",
  #              startDate,endDate,stationName,nullRecords))
  finalDF <- do.call(rbind,rec)
  return(list(data=finalDF,records=length(rec),badRecords=missingNum))
}
 
exploreRawData <- function(stationName, startDate, endDate=startDate) {
  #dates <- seq(as.Date(startDate), as.Date(endDate), by="days")
  #res <- lapply(dates,function(x) {
  #  tmpdf <- getPWSData(stationName,x)
  #  if (is.null(tmpdf)) {return(NULL)}
  #  testData <- select(tmpdf, Time, WindDirectionDegrees, WindSpeedMPH, WindSpeedGustMPH)
  #  return(testData)
  #})
  #nullRecords <- sum(sapply(res,is.null))
  #finalDF <- do.call(rbind,res)

  dataList <-getRawData(stationName, startDate, endDate)
  finalDF <- dataList$data

  print(sprintf("The number of missing day records from %s to %s in %s is %d",
                startDate,endDate,stationName,dataList$badRecords))
  print(summary(finalDF))
  write.csv(finalDF, "test.csv")
  #return(finalDF)
  intervals <- with(finalDF,difftime(Time[-1],Time[-length(Time)]))
  print(head(sort(table(intervals),decreasing=TRUE)))
  print(mean(intervals))
  print(fivenum(intervals))
  #print(finalDF$Time[(finalDF$WindSpeedMPH<0)]) #2013/03/15
  #print(finalDF$Time[(finalDF$WindDirectionDegrees<0)]) #2016/09/20 
  # Sumary doesn't give the info I want
  #print(summary(intervals))
  
  # Timestamps may be duplicated:
  print(finalDF$Time[duplicated(finalDF$Time)])
}

interpolateZoo <- function(myZoo) {

  #return(myZoo)
  resample <- seq(start(myZoo),end(myZoo),by="5 mins")

  emptyZoo <- zoo(,resample)
  mergedZoo <- merge(myZoo,emptyZoo,all=TRUE)
  mergedZoo <- na.approx(mergedZoo)
  mergedZoo <- merge(mergedZoo,emptyZoo,all=FALSE)
  return(mergedZoo)
}
getCleanPWSDataRange <- function(stationName, startDate, endDate) {
  dates <- seq(as.Date(startDate), as.Date(endDate), by="days")
  #badDates <- as.Date(readLines('bad_dates.txt'))
  #dates <- dates[!(dates %in% badDates)]

  dirStr <- sprintf("weatherc/%s",stationName)
  fileName <- sprintf("%s_bad_dates.txt",stationName)
  filePath <- paste(dirStr,fileName,sep="/")
  #print(filePath)
  if( file.exists(filePath) ) {
    badDates <- as.Date(readLines(filePath))
    dates <- dates[!(dates %in% badDates)]
  }
  res <- lapply(dates,function(x) {
    tmpdf <- getPWSData(stationName,x)
    # TODO : get the timezone based on station data
    #tmpdf$Time <- force_tz(tmpdf$Time,"America/Mexico_city")
    #FIXME : the null check probably has to go before the previous line
    if (is.null(tmpdf)) {return(NULL)}
    testData <- select(tmpdf, Time, WindDirectionDegrees, WindSpeedMPH, WindSpeedGustMPH)

    # remove duplicated timestamps: It is possible that duplictes are the next
    # 5 min TS that did not update
    testData <- testData[!duplicated(testData$Time),]
    
    # Negatives become NA:
    testData$WindSpeedMPH[testData$WindSpeedMPH<0]<-NA
    testData$WindDirectionDegrees[testData$WindDirectionDegrees<0]<-NA
    
    # More than 360 degrees is NA:
    testData$WindDirectionDegrees[testData$WindDirectionDegrees>360]<-NA

    # It is mandatory to set the time zone before the interpolation
    # time changes from CST to CDT when that happens
    testData$Time <- force_tz(testData$Time,"America/Mexico_city")   
    

    # Interpolation using Zoo
    myZoo <- convertPWSData2Zoo(testData)
    mergedZoo <- interpolateZoo(myZoo)
    
    testData <- data.frame(Time=time(mergedZoo),coredata(mergedZoo))
    attr(testData$Time, "tzone") <- attr(time(myZoo), "tzone")
    #print(testData$Time)
    
    #print(str(testData$Time))
    #print(resample)  
    
    #2013-03-10 01:05:00
    #2013-03-10 01:05:00
    #2013-03-10 01:15:00
    #2013-03-10 01:15:00
    
    # FIXME: Some extra cleaning can be done here
    # 2016-07-27 has negative on wind gust should be NA
    return(testData)
  })
  #print(dates)
  finalDF <- do.call(rbind,res)
  write.csv(finalDF, "test.csv")
  return(finalDF)
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
#attr(data$dateTime, "tzone") <- "Europe/Paris"

demoZoo <- function() {
  myZoo<- getPWSZoo("IYUCATNT2",as.Date("2014-12-24"))
  zooTimes <- time(myZoo)
  timeDiff <- zooTimes[14]-zooTimes[3]
  print(timeDiff)
  print(timeDiff>hours(1))
  plot(myZoo$WindSpeedMPH)
}

demoWindRose <- function() {
  #dates <- seq(as.Date("2013/1/1"), as.Date("2013/3/31"), by="days")
  dates <- seq(as.Date("2012/1/1"), as.Date("2016/12/31"), by="days")
  badDates <- as.Date(readLines('bad_dates.txt'))
  dates <- dates[!(dates %in% badDates)]
  #dates <- as.Date("2017/6/11") # There is a date with more than 360 deg
  print(dates)
  # Workaround for march 14 and 15 from 2013 which miss the SolarRadiationWatts_m_2
  res <- lapply(dates,function(x) {
    tmpdf <- getPWSData("IYUCATNT2",x)
    #return(tmpdf)
    if (is.null(tmpdf)) {return(NULL)}
    testData <- subset(tmpdf,select=c("Time","WindDirectionDegrees","WindSpeedMPH"))
    return(testData)
    })
  cachedDF <- do.call(rbind,res)
  #testData <- subset(cachedDF,select=c("Time","WindDirectionDegrees","WindSpeedMPH"))
  ren <- rename(cachedDF,date=Time,ws=WindSpeedMPH,wd=WindDirectionDegrees)
  write.csv(ren, "test.csv")
  print(str(ren))
  windRose(ren,cols='heat',type='month',angle=18,paddle=F,ws.int=5,breaks=6,key.footer='mph')
  calendarPlot(ren,pollutant = 'ws',year=2015,annotate='ws')
  # random forest
}

periodLength <- function(measurements,threshVal,threshLength) {
  measurements[is.na(measurements)] <- 0
  rleVec <- rle(measurements>=threshVal)
  contPeriods <- rleVec$lengths[rleVec$values]
  perLen <- sum(contPeriods[contPeriods>=threshLength])
  return(perLen)
}

# TODO: Querry the location from a somewhere but for now hard coded
getPWSLocation <- function(stationName) {
  if (!is.character(stationName)) {
    stop("stationName should be a string")
  }
  if (stationName != "IYUCATNT2") {
    stop(sprintf("Station %s not supported",stationName))
  }
  #Copy paste from web:
  #"lat":"21.341108",
  #"lon":"-89.305756",
  # Original dev:
  #lon <- -89.3
  #lat <- 21.3
  lon <- -89.305756
  lat <- 21.341108
  loc <- matrix(c(lon,lat),nrow=1)
  colnames(loc) <- c("lon","lat")
  return(loc)
}

getPWSMap <- function(stationName) {
  loc <- getPWSLocation(stationName)
  map <- get_map(location="Yucatan",zoom=5)
  plt <- ggmap(map) +
    geom_point(aes(x = lon, y = lat), data = data.frame(loc), alpha = .75, col='red', size=2) +
    labs(title="Location of sensor", x="", y="")
  return(plt)
}

# Worked with this one:
# lon, lat
#loc <- matrix(c(-89.3,21.3),nrow=1)
stationID <- "IYUCATNT2"
loc <- getPWSLocation(stationID)
startDateStr <- "2012/01/01"
endDateStr <- "2016/12/31"

if (TRUE) {
  
#df <- getCleanPWSDataRange(stationID,"2012/03/10","2016/03/10")
#df <- getCleanPWSDataRange(stationID,"2013/04/07","2013/04/07") # DL savings
#df <- getCleanPWSDataRange(stationID,"2012/01/01","2013/12/31")
#df <- getCleanPWSDataRange(stationID,"2016/01/01","2016/01/01")
#df <- getCleanPWSDataRange(stationID,"2012/01/01","2012/01/07") # 2 high winds
#df <- getCleanPWSDataRange(stationID,"2014/01/01","2014/12/31") #2014
#df <- getCleanPWSDataRange(stationID,"2012/01/01","2016/12/31") #Whole

df <- getCleanPWSDataRange(stationID,startDateStr,endDateStr) #Whole

#daySum <- df %>% group_by(date=as.Date(Time,tz=attr(Time,"tzone"))) %>%
#  filter(Time>sunriset(loc, date, direction="sunrise", POSIXct.out=TRUE)[["time"]])
#%>% summarise(num=n())

# Debug timezone comparison
#tmp <- df %>% group_by(date=floor_date(Time,unit="day")) %>%
#  filter(Time>sunriset(loc, date, direction="sunrise", POSIXct.out=TRUE)[["time"]]) %>%
#  filter(Time<sunriset(loc, date, direction="sunset", POSIXct.out=TRUE)[["time"]])
#write.csv(tmp,"test3.csv")

#thresholdPeriod <- dhours(1.5)
thresholdPeriod <- dhours(2)
timeIntervals <- dminutes(5)
thresholdNum <- thresholdPeriod/timeIntervals

# Group by day, extract only daylight readings:
dayLightGroup <- df %>% group_by(date=floor_date(Time,unit="day")) %>%
  filter(Time>sunriset(loc, date, direction="sunrise", POSIXct.out=TRUE)[["time"]]) %>%
  filter(Time<sunriset(loc, date, direction="sunset", POSIXct.out=TRUE)[["time"]])

# summary by day with no filters
daySum <- dayLightGroup  %>%
  summarise(
    avgDlWindSpeedMPH=mean(WindSpeedMPH,na.rm=TRUE),
    avgDlWindDirectionDegrees=mean(WindDirectionDegrees,na.rm=TRUE),
    periodLength15=periodLength(WindSpeedMPH,15,thresholdNum),
    periodLength20=periodLength(WindSpeedMPH,20,thresholdNum),
    periodLength25=periodLength(WindSpeedMPH,25,thresholdNum),
    pseudoWindSpeed=ifelse(periodLength25>0,25,
                           ifelse(periodLength20>0,20,
                                  ifelse(periodLength15>0,15,
                                         0))),
    avgDlWindDirectionDegreesGt15=ifelse(periodLength15>0,
      mean(ifelse(WindSpeedMPH>=15,WindDirectionDegrees,NA),na.rm=TRUE),
      NA),
    avgDlWindSpeedMPHGt15=ifelse(periodLength15>0,
      mean(ifelse(WindSpeedMPH>=15,WindSpeedMPH,NA),na.rm=TRUE),
      NA),
    high=max(WindSpeedMPH,na.rm=TRUE)
    )

# Not important just for debug:
#daySumGt15 <- dayLightGroup  %>%
#  filter(WindSpeedMPH>=15) %>%
#  summarise(
#    avgDlWindSpeedMPHGt15=mean(WindSpeedMPH,na.rm=TRUE),
#    avgDlWindDirectionDegreesGt15=mean(WindDirectionDegrees,na.rm=TRUE)
#  )

#aboveAvg=ifelse(n()>0,mean(WindSpeedMPH),0))

#windRose(daySum,ws="avgDlWindSpeedMPH",wd="avgDlWindDirectionDegrees",cols='heat',angle=10,paddle=FALSE,ws.int=5,breaks=6,key.footer='mph')
calDF <- daySum %>%
  rename(wd=avgDlWindDirectionDegreesGt15) %>%
  mutate(ws=avgDlWindSpeedMPHGt15)
# There is some issue with the calendar and the type of object from dplyr
#calendarPlot(calDF,pollutant="avgDlWindSpeedMPH",annotate='value')
calendarPlot(calDF,pollutant="pseudoWindSpeed",annotate='ws', year=2016)
#calendarPlot(calDF,pollutant="pseudoWindSpeed", year=2016)

# The whole period of time:
#windRose(rename(df,date=Time,ws=WindSpeedMPH,wd=WindDirectionDegrees),cols='heat',angle=10,paddle=FALSE,ws.int=5,breaks=6,key.footer='mph')

#sumarizing by month
monthSum <- daySum %>%
  mutate(month=as.factor(months(date))) %>%
#  rename(target=avgDlWindSpeedMPH) %>%
  rename(target=pseudoWindSpeed) %>%
  group_by(month) %>%
  summarise(
    gteq15=(sum(target>=15,na.rm=TRUE)/n()),
    gteq20=(sum(target>=20,na.rm=TRUE)/n()),
    gteq25=(sum(target>=25,na.rm=TRUE)/n()),
    total=n())


#plt <- ggplot(monthSum) + aes(x=month, fill=gteq15) + geom_bar(position="fill") + scale_y_continuous(labels=percent_format())
#plt <- ggplot(monthSum, aes(x=month, y=gteq15*100, fill = variable)) + geom_bar(stat = 'identity')
#plt <- ggplot(monthSum, aes(x=month, y=gteq15)) + geom_bar(fill="yellow",stat='identity',alpha=0.9) 
#plt <- ggplot(monthSum, aes(x=month)) + geom_bar(y=gteq15,fill="yellow",stat='identity',alpha=0.9) 

plt <- ggplot(NULL, aes(x=month, y=prop)) +
  geom_bar(data=rename(monthSum,prop=gteq15),fill="yellow",stat='identity', width=0.9) +
  geom_bar(data=rename(monthSum,prop=gteq20),fill="orange",stat='identity', width=0.7) +
  geom_bar(data=rename(monthSum,prop=gteq25),fill="red",stat='identity', width=0.5) +
  coord_flip()
print(plt)

#print(df)
write.csv(daySum, "test2.csv")
write.csv(monthSum, "test3.csv")
#sunriset(loc, date, direction="sunrise", POSIXct.out=TRUE)$time

}

