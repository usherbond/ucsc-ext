library(lubridate)
#library(weatherData)
library(zoo)
library(dplyr)
library(openair)
library(maptools)
library(ggplot2)
library(ggmap)

# Gets the station data from the csv cache if it exists or queries the web
getPWSData <- function(stationName,dateQuery,forceWebQuery=FALSE) {
  if (!is.character(stationName)) {
    stop("stationName should be a string")
  }
  if (!is.Date(dateQuery)) {
    stop("dateQuery should be a Date")
  }
  # Extract weather info:
  yearStr <- format(dateQuery,format="%Y")
  monthStr <- format(dateQuery,format="%m")
  # Assuming cache is in the current execution path otherwise we could use ~
  dirStr <- sprintf("weatherc/%s/%s/%s",stationName,yearStr,monthStr)
  # Filename still uses the whole date if we want to flat the dir struct
  fileName <- sprintf("%s-%s.csv",stationName,as.character(dateQuery))
  filePath <- paste(dirStr,fileName,sep="/")
  print(filePath)
  if( file.exists(filePath) && !forceWebQuery ) {
    print("File exists!")
    cachedDF <- read.csv(filePath,stringsAsFactors=FALSE)
    cachedDF <- transform(cachedDF,Time=as.POSIXct(Time))
    return(cachedDF)
  } else {
    print("File missing!")
    if(!dir.exists(dirStr)) {
      dir.create(dirStr,recursive=TRUE)
    }
    weatherDF <- getDetailedWeather(stationName, dateQuery,station_type='id',opt_all_columns=TRUE)
    Sys.sleep(2)
    # Caching code only executed if the force flag is not set
    if (!forceWebQuery && !is.null(weatherDF)) {
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
  #Zoo only seems to work with numbers so removing any character from DF:
  excludeCols <- !(colnames(pwsDF) %in% excludeCols)&(sapply(pwsDF,is.numeric))
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

# Get the data set as a zoo. It won't have non numerics
getPWSZoo <- function(...) {
  return(convertPWSData2Zoo(getPWSData(...)))
}

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
 
# Raw data demo this is a sketch only for the Rmds, only used for debug
exploreRawData <- function(stationName, startDate, endDate=startDate) {
  dataList <-getRawData(stationName, startDate, endDate)
  finalDF <- dataList$data

  print(sprintf("The number of missing day records from %s to %s in %s is %d",
                startDate,endDate,stationName,dataList$badRecords))
  print(summary(finalDF))
  write.csv(finalDF, "test.csv")
  intervals <- with(finalDF,difftime(Time[-1],Time[-length(Time)]))
  print(head(sort(table(intervals),decreasing=TRUE)))
  print(mean(intervals))
  print(fivenum(intervals))
  
  # Timestamps may be duplicated:
  print(finalDF$Time[duplicated(finalDF$Time)])
}

interpolateZoo <- function(myZoo) {
  #1. Create a new empty data set (zoo) with only a 5min time series
  resample <- seq(start(myZoo),end(myZoo),by="5 mins")
  emptyZoo <- zoo(,resample)
  #2. Merge the new empty data set with the old original as a full outer join
  mergedZoo <- merge(myZoo,emptyZoo,all=TRUE)
  #3. Use the operation na.approx to interpolate
  mergedZoo <- na.approx(mergedZoo)
  #4. Merge again the empty dataset with the merged data set as a natural join
  mergedZoo <- merge(mergedZoo,emptyZoo,all=FALSE)
  return(mergedZoo)
}

# Queries the dataset from csv or web and apply cleanup per day plus
# interpolation
getCleanPWSDataRange <- function(stationName, startDate, endDate) {
  dates <- seq(as.Date(startDate), as.Date(endDate), by="days")

  dirStr <- sprintf("weatherc/%s",stationName)
  fileName <- sprintf("%s_bad_dates.txt",stationName)
  filePath <- paste(dirStr,fileName,sep="/")
  if( file.exists(filePath) ) {
    badDates <- as.Date(readLines(filePath))
    dates <- dates[!(dates %in% badDates)]
  }
  res <- lapply(dates,function(x) {
    tmpdf <- getPWSData(stationName,x)
    # TODO : get the timezone based on station data
    #tmpdf$Time <- force_tz(tmpdf$Time,"America/Mexico_city")
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

    return(testData)
  })
  #print(dates)
  finalDF <- do.call(rbind,res)
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
  dates <- seq(as.Date("2012/1/1"), as.Date("2016/12/31"), by="days")
  badDates <- as.Date(readLines('bad_dates.txt'))
  dates <- dates[!(dates %in% badDates)]
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
    labs(title=paste("Location of Station",stationName), x="", y="")
  return(plt)
}

computeDailySummary <- function(wholeDF,location,thresHourPeriod ) {

  thresholdPeriod <- dhours(thresHourPeriod)
  timeIntervals <- dminutes(5)
  thresholdNum <- thresholdPeriod/timeIntervals


  # Group by day, extract only daylight readings:
  dayLightGroup <- df %>% group_by(date=floor_date(Time,unit="day")) %>%
    filter(Time>sunriset(location, date, direction="sunrise", POSIXct.out=TRUE)[["time"]]) %>%
    filter(Time<sunriset(location, date, direction="sunset", POSIXct.out=TRUE)[["time"]])

  # summary by day with no filters
  dailySummary <- dayLightGroup  %>%
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
  return(dailySummary)
}


computeMontlySummary <- function(dailySummary) {
  #sumarizing by month
  monthySummary <- dailySummary %>%
  #  mutate(month=as.factor(months(date))) %>%
    mutate(month=(month(date,label=TRUE,abbr=FALSE))) %>%
  #  rename(target=avgDlWindSpeedMPH) %>%
    rename(target=pseudoWindSpeed) %>%
    group_by(month) %>%
    summarise(
      gteq15=(sum(target>=15,na.rm=TRUE)/n()),
      gteq20=(sum(target>=20,na.rm=TRUE)/n()),
      gteq25=(sum(target>=25,na.rm=TRUE)/n()),
      total=n())
  return(monthySummary)

}


calendarDailySummary <- function(dailySummary, targetYear) {
  calDF <- dailySummary %>%
    rename(wd=avgDlWindDirectionDegreesGt15) %>%
    mutate(ws=avgDlWindSpeedMPHGt15)
  # There is some issue with the calendar and the type of object from dplyr

  calendarPlot(
    calDF,
    pollutant="pseudoWindSpeed",
    annotate='ws',
    year=targetYear,
    breaks=c(0,14,19,24,100),
    labels=c("Poor wind","Low [15,20) mph","Mid [20,25) mph","High 25mph+"),
    main=sprintf("Kiting days in %s",targetDate)
  )
}



ggplotMontlySummary <- function(monthlySummary,startDateString,endDateString) {
  startYear <- year(startDateString)
  endYear <- year(endDateString)
  if (startYear > endYear) {
    stop("Start year must not be greater than end year")
  }
  pltTitle <- "Windy days per month"
  if (startYear == endYear) {
    pltTitle <- paste(pltTitle,sprintf("in %d",startYear))
  } else {
    pltTitle <- paste(pltTitle,sprintf("from %d to %d",startYear,endYear))
  }
  plt <- ggplot(monthlySummary, aes(x=month)) +
    geom_bar(aes(fill="15+ mph",y=gteq15),stat='identity', width=0.9) +
    geom_bar(aes(fill="20+ mph",y=gteq20),stat='identity', width=0.7) +
    geom_bar(aes(fill="25+ mph",y=gteq25),stat='identity', width=0.5) +
    scale_fill_manual("Wind", breaks = c("15+ mph", "20+ mph", "25+ mph"),values = c("#FEBF57", "#F63923", "#800F26")) +
    scale_y_continuous(name="Windy Days (%)", breaks=c(1:10)/10,labels= function(x){return(sprintf("%d%%",x*100))}) +
    labs(title=pltTitle, x="Month") +
    coord_flip()

  return(plt)
}

windRoseCleanData <- function(cleanData) {
  windRose(
    rename(cleanData,ws=WindSpeedMPH,wd=WindDirectionDegrees,date=Time),
    cols='heat',
    type='month',
    angle=18,
    paddle=FALSE,
    ws.int=5,
    breaks=6,
    key.footer='mph')
}





if (TRUE) {
# Worked with this one:
# lon, lat
#loc <- matrix(c(-89.3,21.3),nrow=1)
stationID <- "IYUCATNT2"
loc <- getPWSLocation(stationID)
startDateStr <- "2012/01/01"
#startDateStr <- "2016/01/01"
endDateStr <- "2016/12/31"

startRunTime <- Sys.time() 

#pmap <- getPWSMap(stationID)
#print(pmap)

df <- getCleanPWSDataRange(stationID,startDateStr,endDateStr) #Whole
windRoseCleanData(df)

# Debug timezone comparison
#tmp <- df %>% group_by(date=floor_date(Time,unit="day")) %>%
#  filter(Time>sunriset(loc, date, direction="sunrise", POSIXct.out=TRUE)[["time"]]) %>%
#  filter(Time<sunriset(loc, date, direction="sunset", POSIXct.out=TRUE)[["time"]])
#write.csv(tmp,"test3.csv")

daySum <- computeDailySummary(df,loc,2)

# Not important just for debug:
#daySumGt15 <- dayLightGroup  %>%
#  filter(WindSpeedMPH>=15) %>%
#  summarise(
#    avgDlWindSpeedMPHGt15=mean(WindSpeedMPH,na.rm=TRUE),
#    avgDlWindDirectionDegreesGt15=mean(WindDirectionDegrees,na.rm=TRUE)
#  )


targetDate <- year(endDateStr)
calendarDailySummary(daySum,targetDate)

#sprintf("Kiting days in %s",targetDate)

#print(year(endDateStr))

#calendarPlot(calDF,pollutant="pseudoWindSpeed", year=2016)

# The whole period of time:
#windRose(rename(df,date=Time,ws=WindSpeedMPH,wd=WindDirectionDegrees),cols='heat',angle=10,paddle=FALSE,ws.int=5,breaks=6,key.footer='mph')

monthSum  <- computeMontlySummary(daySum)

plt <- ggplotMontlySummary(monthSum,startDateStr,endDateStr)
print(plt)


#print(df)
write.csv(df, "test.csv")
write.csv(daySum, "test2.csv")
write.csv(monthSum, "test3.csv")
#sunriset(loc, date, direction="sunrise", POSIXct.out=TRUE)$time

diffRunTime <- Sys.time() - startRunTime
print(diffRunTime)
}

