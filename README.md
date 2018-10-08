# TrafficAPI - API interface to the traffic Datalake
This is a sample python based HTTP interface to the datalake traffic API that support operation on traffic datalake. 

## TrafficAPI helps you to:

* store file traffic gathered from censor to data lake
* store file manchester united football matches crawled from website "http://www.worldfootball.net/teams/manchester-united/"
* integrate data matches and traffics 
* perform outlier detection using arima algorithm
* Perform outlier detection using lstmnrn
* Perform outlier detection using extracs

## Available APIs

#### POST `/traffic`

You can do a POST to `/traffics` to put traffic file into datalake

The body must have:

* `file`: name of the file 

It returns the following:

```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>report traffic api</title>
 
</head>
<body>
    <div class="row">
            Processing Time : 1.73                 
        <br>
        <br>
    </div>
    <div class="table">                
            Traffic file successfuly saved  
    </div>
    <div class="image">
    </div>
</body>
</html>
```


#### POST `/match`

You can do a POST to `/match`,web scrapping from "http://www.worldfootball.net/teams/manchester-united/" to datalake.

The body must have:

* `year`: the year

It returns the following:

```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>report traffic api</title>
 
</head>
<body>
    <div class="row">
            Processing Time : 1.73
        <br>
        <br>
    </div>
    <div class="table">
            Match file successfuly saved  
    </div>
    <div class="image">
    </div>
</body>
</html>
```

##### GET `/integration/<monthfrom>/<yearfrom>/<monthto>/<yearto>/<place>/<int:bfrduration>/<int:afterduration>`

You can perform data traffic and data matches integration, 

The body must have:

monthfrom : name of the month (Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dev)

yearfrom : 4 digit year (etc: 2012,2014..)

monthto : name of the month (Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dev)

yearto : 4 digit year (etc: 2012,2014..)

place : home or away

bfrduration : length of duration before the kick off in hour 

afterduration : length of duration after the match in hour


It returns the following:

```
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>report traffic api</title>
    </head>
    <body>
        <div class="row">
        
    
            Processing Time : 8.04 seconds
                 
        
            <br>
            <br>
    
        </div>
        <div class="table">
                
            Sdate,Cosit,LaneNumber,LaneDescription,LaneDirection,DirectionDescription,Volume,Flags,Flag Text,AvgSpeed,PmlHGV,Class1Volume,Class2Volume,Class3Volume,Class4Volume,Class5Volume,Class6Volume,Outlier,Date,Hour,DayNumber,DayName,Week
2012-11-01 00:00:00,1083,1,NB_NS,1,North,50,0,,17711,20,0,49,0,1,0,0,0,2012/11/01,00,3,Thursday,Weekdays
2012-11-01 00:00:00,1083,2,NB_MID,1,North,94,0,,18227,32,0,91,0,2,1,0,0,2012/11/01,00,3,Thursday,Weekdays
2012-11-01 00:00:00,1083,3,NB_OS,1,North,26,0,,18526,0,1,25,0,0,0,0,0,2012/11/01,00,3,Thursday,Weekdays
```

### Quotes API

#### GET `/arima`

It returns a String with a Random quote from Chuck Norris. It doesn't require authentication.

#### GET `/lstmrnn`

It returns a String with a Random quote from Chuck Norris. It requires authentication. 

The JWT - `access_token` must be sent on the `Authorization` header as follows: `Authorization: Bearer {jwt}`

#### GET `/extracs`

It returns processing time, balanced accuracy and  lists of outliers
The body must have:

* `trainfile`: name of the training file 
* `testfile`: name of the testing file 

It returns the following:
```
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>report traffic api</title>
    </head>
    <body>
        <div class="row">
        
    
            Processing Time : 399.6 seconds
                 
        
            <br>
            <br>
    
        </div>
        <div class="table">
            Balanced accuracy: 0.9816722972972973
    
            
            <table border="1" class="dataframe">
                <thead>
                    <tr style="text-align: right;">
                        <th></th>
                        <th>Specified</th>
                        <th>Condition</th>
                        <th>Phenotype</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <th>3</th>
                        <td>[2, 1, 3, 4, 0]</td>
                        <td>[[512.375, 835.625], '2', '21', '1', '5']</td>
                        <td>1</td>
                    </tr>
                    <tr>
                        <th>4</th>
                        <td>[2, 1]</td>
                        <td>[[25.97999999999999, 388.02], '2']</td>
                        <td>1</td>
                    </tr>
```

## Running it

Just clone the repository, install library needed and run `python trafficapi.py`. That's it :).

Library 

## Issue Reporting

If you have found a bug or if you have a feature request, please report them at this repository issues section. Please do not report security vulnerabilities on the public GitHub issue tracker. The [Responsible Disclosure Program](https://auth0.com/whitehat) details the procedure for disclosing security issues.

## Author

[Auth0](https://auth0.com)

## License

This project is licensed under the MIT license. See the [LICENSE](LICENSE) file for more info.
## What is Auth0?



## Create a free account in Auth0

1. Go to [Auth0](https://auth0.com) and click Sign Up.
2. Use Google, GitHub or Microsoft Account to login.

## Use Postman

Postman provides a powerful GUI platform to make your API development faster & easier, from building API requests through testing, documentation and sharing

Here is a [small collection](https://documenter.getpostman.com/view/3232248/auth0-nodejs-jwt-auth/7LnAi4o) to highlight the features of this sample API.

[![Run NodeJS JWT Authentication in Postman](https://run.pstmn.io/button.svg)](https://app.getpostman.com/run-collection/c57ddc507592c436662c)
**Awesome web-browsable Web APIs.**
