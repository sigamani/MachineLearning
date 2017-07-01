import os
import pandas as pd
import pyodbc
import calendar
import csv


engine = pyodbc.connect('Driver={SQL Server};Server=10.0.60.19;Database=SRPA_ENT_Master;Uid=sa;Pwd=GobStopper;')



def makeSplitFiles():

    for s in range(0,2):

        for m in range(1,4):

            #year, startmonth, countryID, returnVal
            query = sp.getFactorData(2016,m,59,s)
            df = pd.read_sql_query(query, engine)

            if (m==1): fname = "JanFeb"
            if (m==2): fname = "FebMar"
            if (m==3): fname = "MarApr"
            if (s==0): signal = "B"
            if (s==1): signal = "S"

            filename = r"C:\\Users\\michael\\"+fname+"2016"+signal+".data"
            df.to_csv(filename, index=False)




def makeCombinedFiles():

        for m in range(1,4):

            #year, startmonth, countryID
            query = getFactorData2(2016,m,59)
            df = pd.read_sql_query(query, engine)

            if (m==1): fname = "JanFeb"
            if (m==2): fname = "FebMar"
            if (m==3): fname = "MarApr"

            filename = r"C:\\Users\\michael\\"+fname+"2016.data"

            #fill nans with mean of column
            df2 = df.fillna(df.mean())
            df2.to_csv(filename, index=False)
            #print(df2)





def getFactorData(year, startmonth,countryID,returnVal):
      
    fromMonthNDays = calendar.monthrange(year,startmonth)[1]
    fromMonthNDays2 = calendar.monthrange(year,startmonth+1)[1]
    fromMonthNDays3 = calendar.monthrange(year,startmonth+2)[1]

    fromDate = str(year) + '-' + str(startmonth) + '-' + str(fromMonthNDays)
    toDate = str(year)+ '-' + str(startmonth+1) + '-' + str(fromMonthNDays2)

    returnfromDate = str(year)+ '-' + str(startmonth+1) + '-' + str(fromMonthNDays2)
    returntoDate = str(year)+ '-' + str(startmonth+2) + '-' + str(fromMonthNDays3)
   
    string = """

	  select * from 
	  (	
      select
      --b2p.securityId as [SecurityID],
      case when b2p.value <> 0 then b2p2.value/b2p.value -1 else null end as BookToPrice,
      case when dy.value <> 0 then dy2.value/dy.value -1 else null end as DividendYield,
      case when ey.value <> 0 then ey2.value/ey.value -1 else null end as EarningsYield,
      case when sg.value <> 0 then sg2.value/sg.value -1 else null end as SalesGrowth,
      case when a2e.value <> 0 then a2e2.value/a2e.value -1 else null end as AssetsToEquity,
      case when mc.value <> 0 then mc2.value/mc.value -1 else null end as MarketCap,
      case when mb.value <> 0 then mb2.value/mb.value -1 else null end as MarketBeta,
      case when d2e.value <> 0 then d2e2.value/d2e.value -1 else null end as DebtToEquity,
      case when v1yr.value <> 0 then v1yr2.value/v1yr.value -1 else null end as [1YrVol],
      case when v5yr.value <> 0 then v5yr2.value/v5yr.value -1 else null end as [5YrVol],
      case when v3yr.value <> 0 then v3yr2.value/v3yr.value -1 else null end as [3YrVol],
      case when e2cg.value <> 0 then e2cg2.value/e2cg.value -1 else null end as ExposureToCurrencyGain,
      case when stM.value <> 0 then stM2.Value/stM.Value -1 else null end as MomentumST,
      case when ltM.value <> 0 then ltM2.Value/ltM.Value -1 else null end as MomentumLT,
      case when sentm.value <> 0 then sentm2.Value/sentm.Value -1 else null end as Sentiment,
      case when roe.value <> 0 then roe2.Value/roe.Value -1 else null end as ReturnOnEquity,
      case when tat.value <> 0 then tat2.Value/tat.Value -1 else null end as TotalAssetTurnover,
      z.SectorID,

      case when rtn.value <> 0 and rtn2.value/rtn.value > 0 then 1 else 0 end as MonthlyReturn

      from
      (select * from factorData
      where factorID = 20
      and factorDate = '""" + returnfromDate + """') rtn
      join
      (select * from factorData
      where factorID = 20
      and factorDate = '""" + returntoDate + """') rtn2 on rtn.securityID = rtn2.securityID
      join
      (select * from factorData
      where factorID = 1
      and factorDate = '""" + fromDate + """') b2p on rtn.securityID = b2p.securityID
      join
      (select * from factorData
      where factorID = 1
      and factorDate = '""" + toDate + """') b2p2 on rtn.securityID = b2p2.securityID
      join
      (select * from factorData
      where factorID = 2
      and factorDate = '""" + fromDate + """') dy on rtn.securityID = dy.securityID
      join
      (select * from factorData
      where factorID = 2
      and factorDate = '""" + toDate + """') dy2 on rtn.securityID = dy2.securityID
      join
      (select * from factorData
      where factorID = 3
      and factorDate = '""" + fromDate + """') ey on rtn.securityID = ey.securityID
      join
      (select * from factorData
      where factorID = 3
      and factorDate = '""" + toDate + """') ey2 on rtn.securityID = ey2.securityID
      join
      (select * from factorData
      where factorID = 10
      and factorDate = '""" + fromDate + """') sg on rtn.securityID = sg.securityID
      join
      (select * from factorData
      where factorID = 10
      and factorDate = '""" + toDate + """') sg2 on rtn.securityID = sg2.securityID
      join
      (select * from factorData
      where factorID = 112
      and factorDate = '""" + fromDate + """') a2e on rtn.securityID = a2e.securityID
      join
      (select * from factorData
      where factorID = 112
      and factorDate = '""" + toDate + """') a2e2 on rtn.securityID = a2e2.securityID
      join
      (select * from factorData
      where factorID = 13
      and factorDate = '""" + fromDate + """') mc on rtn.securityID = mc.securityID
      join
      (select * from factorData
      where factorID = 13
      and factorDate = '""" + toDate + """') mc2 on rtn.securityID = mc2.securityID
      join
      (select * from factorData
      where factorID = 14
      and factorDate = '""" + fromDate + """') mb on rtn.securityID = mb.securityID
      join
      (select * from factorData
      where factorID = 14
      and factorDate = '""" + toDate + """') mb2 on rtn.securityID = mb2.securityID
      join
      (select * from factorData
      where factorID = 17
      and factorDate = '""" + fromDate + """') d2e on rtn.securityID = d2e.securityID
      join
      (select * from factorData
      where factorID = 17
      and factorDate = '""" + toDate + """') d2e2 on rtn.securityID = d2e2.securityID
      join
      (select * from factorData
      where factorID = 116
      and factorDate = '""" + fromDate + """') v1yr on rtn.securityID = v1yr.securityID
      join
      (select * from factorData
      where factorID = 116
      and factorDate = '""" + toDate + """') v1yr2 on rtn.securityID = v1yr2.securityID
      join
      (select * from factorData
      where factorID = 115
      and factorDate = '""" + fromDate + """') v5yr on rtn.securityID = v5yr.securityID
      join
      (select * from factorData
      where factorID = 115
      and factorDate = '""" + toDate + """') v5yr2 on rtn.securityID = v5yr2.securityID
      join
      (select * from factorData
      where factorID = 111
      and factorDate = '""" + fromDate + """') v3yr on rtn.securityID = v3yr.securityID
      join
      (select * from factorData
      where factorID = 111
      and factorDate = '""" + toDate + """') v3yr2 on rtn.securityID = v3yr2.securityID
      join
      (select * from factorData
      where factorID = 56
      and factorDate = '""" + fromDate + """') e2cg on rtn.securityID = e2cg.securityID
      join
      (select * from factorData
      where factorID = 56
      and factorDate = '""" + toDate + """') e2cg2 on rtn.securityID = e2cg2.securityID
      join
      (select * from factorData
      where factorID = 7
      and factorDate = '""" + fromDate + """') roe on rtn.securityID = roe.securityID
      join
      (select * from factorData
      where factorID = 7
      and factorDate = '""" + toDate + """') roe2 on rtn.securityID = roe2.securityID
      join
      (select * from factorData
      where factorID = 101
      and factorDate = '""" + fromDate + """') tat on rtn.securityID = tat.securityID
      join
      (select * from factorData
      where factorID = 101
      and factorDate = '""" + toDate + """') tat2 on rtn.securityID = tat2.securityID
      join
      (select * from SecuritySector
      where ClassificationTypeID = 1) z on rtn.securityID = z.securityID
      join
      ( select * from securityCountry
      where classificationtypeid = 1
      ) sc on rtn.securityID = sc.securityID
      join
      (select SecurityID, 0.067837311762143 as [S&P500 Bef], 0.0038780205054853 as [S&P500 Aft]
      from factorData
      where factorID = 56
      and factorDate = '""" + toDate + """') snp on rtn.SecurityID = snp.SecurityID
      join
      (select * from factorData
      where factorID = 15
      and factorDate = '""" + fromDate + """') stM on rtn.securityID = stM.securityID
      join
      (select * from factorData
      where factorID = 15
      and factorDate = '""" + toDate + """') stM2 on rtn.securityID = stM2.securityID
      join
      (select * from factorData
      where factorID = 16
      and factorDate = '""" + fromDate + """') ltM on rtn.securityID = ltM.securityID
      join
      (select * from factorData
      where factorID = 16
      and factorDate = '""" + toDate + """') ltM2 on rtn.securityID = ltM2.securityID
      join
      (select * from factorData
      where factorID = 81
      and factorDate = '""" + fromDate + """') sentm on rtn.securityID = sentm.securityID
      join
      (select * from factorData
      where factorID = 81
      and factorDate = '""" + toDate + """') sentm2 on rtn.securityID = sentm2.securityID

      where CountryID = """ + str(countryID) + """
	  ) tab 

	  where tab.MonthlyReturn= """ + str(returnVal) + """
      """

    return string


def getFactorData2(year, startmonth,countryID):
      
    fromMonthNDays = calendar.monthrange(year,startmonth)[1]
    fromMonthNDays2 = calendar.monthrange(year,startmonth+1)[1]
    fromMonthNDays3 = calendar.monthrange(year,startmonth+2)[1]

    fromDate = str(year) + '-' + str(startmonth) + '-' + str(fromMonthNDays)
    toDate = str(year)+ '-' + str(startmonth+1) + '-' + str(fromMonthNDays2)

    returnfromDate = str(year)+ '-' + str(startmonth+1) + '-' + str(fromMonthNDays2)
    returntoDate = str(year)+ '-' + str(startmonth+2) + '-' + str(fromMonthNDays3)
   
    string = """


      select
      --b2p.securityId as [SecurityID],
      case when b2p.value <> 0 then b2p2.value/b2p.value -1 else null end as BookToPrice,
      case when dy.value <> 0 then dy2.value/dy.value -1 else null end as DividendYield,
      case when ey.value <> 0 then ey2.value/ey.value -1 else null end as EarningsYield,
      case when sg.value <> 0 then sg2.value/sg.value -1 else null end as SalesGrowth,
      case when a2e.value <> 0 then a2e2.value/a2e.value -1 else null end as AssetsToEquity,
      case when mc.value <> 0 then mc2.value/mc.value -1 else null end as MarketCap,
      case when mb.value <> 0 then mb2.value/mb.value -1 else null end as MarketBeta,
      case when d2e.value <> 0 then d2e2.value/d2e.value -1 else null end as DebtToEquity,
      case when v1yr.value <> 0 then v1yr2.value/v1yr.value -1 else null end as [1YrVol],
      case when v5yr.value <> 0 then v5yr2.value/v5yr.value -1 else null end as [5YrVol],
      case when v3yr.value <> 0 then v3yr2.value/v3yr.value -1 else null end as [3YrVol],
      case when e2cg.value <> 0 then e2cg2.value/e2cg.value -1 else null end as ExposureToCurrencyGain,
      case when stM.value <> 0 then stM2.Value/stM.Value -1 else null end as MomentumST,
      case when ltM.value <> 0 then ltM2.Value/ltM.Value -1 else null end as MomentumLT,
      case when sentm.value <> 0 then sentm2.Value/sentm.Value -1 else null end as Sentiment,
      case when roe.value <> 0 then roe2.Value/roe.Value -1 else null end as ReturnOnEquity,
      case when tat.value <> 0 then tat2.Value/tat.Value -1 else null end as TotalAssetTurnover,
      z.SectorID,

      case when rtn.value <> 0 and rtn2.value/rtn.value > 0 then 1 else 0 end as MonthlyReturn

      from
      (select * from factorData
      where factorID = 20
      and factorDate = '""" + returnfromDate + """') rtn
      join
      (select * from factorData
      where factorID = 20
      and factorDate = '""" + returntoDate + """') rtn2 on rtn.securityID = rtn2.securityID
      join
      (select * from factorData
      where factorID = 1
      and factorDate = '""" + fromDate + """') b2p on rtn.securityID = b2p.securityID
      join
      (select * from factorData
      where factorID = 1
      and factorDate = '""" + toDate + """') b2p2 on rtn.securityID = b2p2.securityID
      join
      (select * from factorData
      where factorID = 2
      and factorDate = '""" + fromDate + """') dy on rtn.securityID = dy.securityID
      join
      (select * from factorData
      where factorID = 2
      and factorDate = '""" + toDate + """') dy2 on rtn.securityID = dy2.securityID
      join
      (select * from factorData
      where factorID = 3
      and factorDate = '""" + fromDate + """') ey on rtn.securityID = ey.securityID
      join
      (select * from factorData
      where factorID = 3
      and factorDate = '""" + toDate + """') ey2 on rtn.securityID = ey2.securityID
      join
      (select * from factorData
      where factorID = 10
      and factorDate = '""" + fromDate + """') sg on rtn.securityID = sg.securityID
      join
      (select * from factorData
      where factorID = 10
      and factorDate = '""" + toDate + """') sg2 on rtn.securityID = sg2.securityID
      join
      (select * from factorData
      where factorID = 112
      and factorDate = '""" + fromDate + """') a2e on rtn.securityID = a2e.securityID
      join
      (select * from factorData
      where factorID = 112
      and factorDate = '""" + toDate + """') a2e2 on rtn.securityID = a2e2.securityID
      join
      (select * from factorData
      where factorID = 13
      and factorDate = '""" + fromDate + """') mc on rtn.securityID = mc.securityID
      join
      (select * from factorData
      where factorID = 13
      and factorDate = '""" + toDate + """') mc2 on rtn.securityID = mc2.securityID
      join
      (select * from factorData
      where factorID = 14
      and factorDate = '""" + fromDate + """') mb on rtn.securityID = mb.securityID
      join
      (select * from factorData
      where factorID = 14
      and factorDate = '""" + toDate + """') mb2 on rtn.securityID = mb2.securityID
      join
      (select * from factorData
      where factorID = 17
      and factorDate = '""" + fromDate + """') d2e on rtn.securityID = d2e.securityID
      join
      (select * from factorData
      where factorID = 17
      and factorDate = '""" + toDate + """') d2e2 on rtn.securityID = d2e2.securityID
      join
      (select * from factorData
      where factorID = 116
      and factorDate = '""" + fromDate + """') v1yr on rtn.securityID = v1yr.securityID
      join
      (select * from factorData
      where factorID = 116
      and factorDate = '""" + toDate + """') v1yr2 on rtn.securityID = v1yr2.securityID
      join
      (select * from factorData
      where factorID = 115
      and factorDate = '""" + fromDate + """') v5yr on rtn.securityID = v5yr.securityID
      join
      (select * from factorData
      where factorID = 115
      and factorDate = '""" + toDate + """') v5yr2 on rtn.securityID = v5yr2.securityID
      join
      (select * from factorData
      where factorID = 111
      and factorDate = '""" + fromDate + """') v3yr on rtn.securityID = v3yr.securityID
      join
      (select * from factorData
      where factorID = 111
      and factorDate = '""" + toDate + """') v3yr2 on rtn.securityID = v3yr2.securityID
      join
      (select * from factorData
      where factorID = 56
      and factorDate = '""" + fromDate + """') e2cg on rtn.securityID = e2cg.securityID
      join
      (select * from factorData
      where factorID = 56
      and factorDate = '""" + toDate + """') e2cg2 on rtn.securityID = e2cg2.securityID
      join
      (select * from factorData
      where factorID = 7
      and factorDate = '""" + fromDate + """') roe on rtn.securityID = roe.securityID
      join
      (select * from factorData
      where factorID = 7
      and factorDate = '""" + toDate + """') roe2 on rtn.securityID = roe2.securityID
      join
      (select * from factorData
      where factorID = 101
      and factorDate = '""" + fromDate + """') tat on rtn.securityID = tat.securityID
      join
      (select * from factorData
      where factorID = 101
      and factorDate = '""" + toDate + """') tat2 on rtn.securityID = tat2.securityID
      join
      (select * from SecuritySector
      where ClassificationTypeID = 1) z on rtn.securityID = z.securityID
      join
      ( select * from securityCountry
      where classificationtypeid = 1
      ) sc on rtn.securityID = sc.securityID
      join
      (select SecurityID, 0.067837311762143 as [S&P500 Bef], 0.0038780205054853 as [S&P500 Aft]
      from factorData
      where factorID = 56
      and factorDate = '""" + toDate + """') snp on rtn.SecurityID = snp.SecurityID
      join
      (select * from factorData
      where factorID = 15
      and factorDate = '""" + fromDate + """') stM on rtn.securityID = stM.securityID
      join
      (select * from factorData
      where factorID = 15
      and factorDate = '""" + toDate + """') stM2 on rtn.securityID = stM2.securityID
      join
      (select * from factorData
      where factorID = 16
      and factorDate = '""" + fromDate + """') ltM on rtn.securityID = ltM.securityID
      join
      (select * from factorData
      where factorID = 16
      and factorDate = '""" + toDate + """') ltM2 on rtn.securityID = ltM2.securityID
      join
      (select * from factorData
      where factorID = 81
      and factorDate = '""" + fromDate + """') sentm on rtn.securityID = sentm.securityID
      join
      (select * from factorData
      where factorID = 81
      and factorDate = '""" + toDate + """') sentm2 on rtn.securityID = sentm2.securityID

      where CountryID = """ + str(countryID) + """
      """

    return string


if __name__ == "__main__":

    makeCombinedFiles()