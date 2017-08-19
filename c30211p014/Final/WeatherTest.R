library(lubridate)
library(weatherData)
library(zoo)
library(dplyr)
library(openair)
library(maptools)

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
    Sys.sleep(5)
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
    # FIXME : get the timezone based on station data
    tmpdf$Time <- force_tz(tmpdf$Time,"America/Mexico_city")
    if (is.null(tmpdf)) {return(NULL)}
    testData <- select(tmpdf, Time, WindDirectionDegrees, WindSpeedMPH, WindSpeedGustMPH)
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

#gtEqNum <- function(data,threshold){
#  return(sum(data>=threshold))
#}

loc <- matrix(c(-89.3,21.3),nrow=1)
df <- getCleanPWSDataRange("IYUCATNT2","2014/01/01","2014/12/31")
#dfsum <- df %>% group_by(date=as.Date(Time,tz=attr(Time,"tzone"))) %>%
#  filter(Time>sunriset(loc, date, direction="sunrise", POSIXct.out=TRUE)[["time"]])
#%>% summarise(num=n())

dfsum <- df %>% group_by(date=floor_date(Time,unit="day")) %>%
  filter(Time>sunriset(loc, date, direction="sunrise", POSIXct.out=TRUE)[["time"]]) %>%
  filter(Time<sunriset(loc, date, direction="sunset", POSIXct.out=TRUE)[["time"]]) %>%
  summarise(avgDlWindSpeedMPH=mean(WindSpeedMPH),avgDlWindDirectionDegrees=mean(WindDirectionDegrees),high=max(WindSpeedMPH))


#windRose(dfsum,ws="avgDlWindSpeedMPH",wd="avgDlWindDirectionDegrees",cols='heat',angle=10,paddle=FALSE,ws.int=5,breaks=6,key.footer='mph')
calDF <- dfsum %>%
  rename(wd=avgDlWindDirectionDegrees) %>%
  mutate(ws=avgDlWindSpeedMPH)
# There is some issue with the calendar and the type of object from dplyr
calendarPlot(calDF,year=2014,pollutant="avgDlWindSpeedMPH",annotate='ws')
# The whole period of time:
#windRose(rename(df,date=Time,ws=WindSpeedMPH,wd=WindDirectionDegrees),cols='heat',angle=10,paddle=FALSE,ws.int=5,breaks=6,key.footer='mph')

#sumarizing by month
monthSum <- dfsum %>%
  mutate(month=as.factor(months(date))) %>%
  rename(target=avgDlWindSpeedMPH) %>%
  group_by(month) %>%
  summarise(
    gteq15=(sum(target>=15)/n()),
    gteq20=(sum(target>=20)/n()),
    gteq25=(sum(target>=25)/n()),
    total=n())

library(ggplot2)
#plt <- ggplot(monthSum) + aes(x=month, fill=gteq15) + geom_bar(position="fill") + scale_y_continuous(labels=percent_format())
#plt <- ggplot(monthSum, aes(x=month, y=gteq15*100, fill = variable)) + geom_bar(stat = 'identity')
#plt <- ggplot(monthSum, aes(x=month, y=gteq15)) + geom_bar(fill="yellow",stat='identity',alpha=0.9) 
#plt <- ggplot(monthSum, aes(x=month)) + geom_bar(y=gteq15,fill="yellow",stat='identity',alpha=0.9) 

plt <- ggplot(NULL, aes(x=month, y=prop)) +
  geom_bar(data=rename(monthSum,prop=gteq15),fill="yellow",stat='identity', width=0.9) +
  geom_bar(data=rename(monthSum,prop=gteq20),fill="orange",stat='identity', width=0.7) +
  geom_bar(data=rename(monthSum,prop=gteq25),fill="red",stat='identity', width=0.5) +
  coord_flip()


#print(df)
write.csv(dfsum, "test2.csv")
#sunriset(loc, date, direction="sunrise", POSIXct.out=TRUE)$time
