import numpy as np
import pandas as pd
import csv as cv
import sys
import scipy as sp
import xlrd as xlr
import operator


NFACTORS = 84
THRESHOLD = 0.01
ITERATIONS = 100 #100


factorList = ['Econ - Exposure to Currency Gain',
'Econ - Exposure to GDP Surprise',
'Econ - Exposure to Gold Return',
'Econ - Exposure to Inflation',
'Econ - Exposure to Oil Return',
'Econ - Exposure to Short Rate',
'Env - - Climate Change - MSCI IVA',
'Env - - Environmental - GMI',
'Env - - Environmental - oekom']


factorList = ['Econ - Exposure to Currency Gain',
'Econ - Exposure to GDP Surprise',
'Econ - Exposure to Gold Return',
'Econ - Exposure to Inflation',
'Econ - Exposure to Oil Return',
'Econ - Exposure to Short Rate',
'Fin - Assets to Equity',
'G - 3 Year Asset Growth',
'G - 3 Year Capex Growth',
'G - 5 Year Asset Growth',
'G - 5 Year Capex Growth',
'G - 5 Year Div Growth',
'G - Earnings Growth 5 year',
'G - Gross Profit Margin',
'G - Gross Profits to Assets',
'G - Growth in Earnings per Share',
'G - IBES 12M Rev (1M)',
'G - IBES 12M Rev (3M)',
'G - IBES Earnings 12 Month Growth F\'casts',
'G - IBES Earnings Long Term Growth F\'casts',
'G - IBES FY1 Earnings F\'cast Rev\'s 1M Sample',
'G - IBES FY1 Earnings F\'cast Rev\'s 3M Sample',
'G - IBES FY2 Earnings F\'cast Rev\'s 1M Sample',
'G - IBES FY2 Earnings F\'cast Rev\'s 3M Sample',
'G - IBES ROE',
'G - IBES Sales 12 Month Growth F\'casts',
'G - IBES Sales Long Term Growth F\'casts',
'G - Income to Sales (Profit Margin)',
'G - Operating Profit Margin',
'G - Return on Assets',
'G - Return on Equity',
'G - Return on Invested Capital',
'G - Sales Growth',
'G - Sales Growth 5 year',
'G - Sustainable Growth Rate',
'M - Momentum 12 Month',
'M - Momentum 12-1',
'M - Momentum 6 Month',
'Misc - Asset Turnover',
'Misc - Current Ratio (ex-fin)',
'Misc - Dividend Payout Ratio',
'Misc - Int Coverage Ratio (ex-fin)',
'Misc - Quick Ratio (ex-fin)',
'Misc - Trading Turnover 3M',
'Q - Earnings Gr Stability 5yr',
'Q - Low Accruals',
'Q - Sales Gr Stability 5yr',
'Q - Stability of Earnings Growth',
'Q - Stability of IBES 12M Earnings Gr F\'casts',
'Q - Stability of IBES FY1 Earnings F\'cast Rev\'s',
'Q - Stability of Returns',
'Q - Stability of Sales Growth',
'Rsk - Beta',
'Rsk - Daily Volatility (1 Yr)',
'Rsk - Debt to Equity',
'Rsk - Foreign Sales as % of Total Sales',
'Rsk - Volatility 1 Year',
'Rsk - Volatility 3 Year',
'Rsk - Volatility 5 Year',
'S - Market Cap',
'S - Total Book Value',
'S - Total Earnings',
'S - Total Number of Employees',
'S - Total Sales',
'V - Book Value per Share to Price',
'V - Cashflow per Share to Price',
'V - Cyc Adj Engs Yld',
'V - Dividend Yield',
'V - Earnings per Share to Price',
'V - EBIT to EV',
'V - EBITDA to EV',
'V - EBITDA to Price',
'V - Free Cashflow Yield',
'V - IBES F\'cast Dividend Yield',
'V - IBES F\'cast Earnings Yield',
'V - IBES F\'cast Sales Yield',
'V - Inverse PEG',
'V - Inverse PEGY',
'V - Net Buyback Yield',
'V - Net Debt Paydown Yield',
'V - Net Payout Yield',
'V - Sales per Share to Price',
'V - Sales to EV',
'V - Total Shareholder Yield']




      


def makeReturnSeriesMatrix0(sheet):

    ReturnSeries = [] 
    for j in range(NFACTORS):
        
        list = []
      #  for i in range(181,241):  #last 60M
      #  for i in range(120,180):  #last 60M
      #  for i in range(119,179):  #last 60M
        for i in range(58,118):  #last 60M

            value = sheet.cell_value(j+1,i)
            list.append(value)

        ReturnSeries.append(list)

    return ReturnSeries




def makeReturnSeriesMatrix1(m, list, counter):

    newList2 = []

    lList = len(list) 
    


    for i in range(0,60):
    
        val = 0
        for j in range(lList):

            val += m[list[j]][i]
          
        newList2.append( val ) 


    count = counter[list[0]] + counter[list[1]]

    m.append(newList2)
    counter.append(count)
    return m





def getCorrelation(m, iteration):

    results = []
    for k in range(NFACTORS+iteration):

        results.append([])
        for l in range(NFACTORS+iteration):
                    
            corr = np.corrcoef(m[k],m[l])
            temp = corr[0,1]    
            results[k].append(temp)

    return results  


def getResult0(m, iteration):

    corrs = getCorrelation(m, iteration)    

    max = 0
    imax = 0
    jmax=0

    for i in range(NFACTORS + iteration):

        for j in range(NFACTORS + iteration):

            if (i == j ): continue           
            if (corrs[i][j] > max):

                max = corrs[i][j]
                imax = i
                jmax = j
    

    print( "%f, %s, %s" % ( max, factorList[imax], factorList[jmax]) )               
    return (max, imax, jmax)   




def getResult1(m, list, iteration, counter):

    newM = []

    for i in range(NFACTORS+iteration):
        newM.append([])

        for j in range(0,60):
            newM[i].append(m[i][j]/counter[i])

  

    corrs = getCorrelation(newM, iteration)    

    max = 0
    jmax=0
    imax=0  
    

    for i in range(NFACTORS+iteration):

        if i not in list:

            for j in range(NFACTORS+iteration):
        
                if j not in list:
 
                    if (i == j): continue  
                    if (corrs[i][j] > max):

                        max = corrs[i][j]
                        imax = i
                        jmax = j
    

    print( "%f, %s, %s" % ( max, factorList[imax], factorList[jmax]) )               
    return (max, imax, jmax)   




if __name__ == "__main__":

    dir = r"C:\\Users\\michael\\Downloads\\"
    file = dir + "Dev 1500 CASA2.xlsx"
    book=xlr.open_workbook(file)                         
    sheet=book.sheet_by_name('Rel returns')  

    m0 = makeReturnSeriesMatrix0(sheet)
    r0 = getResult0(m0, 0) # returns tuple correlation with name of factor
    list = []
    list2= []
   
    list.append(r0[1])
    list.append(r0[2])

    list2.append(r0[1]) ##
    list2.append(r0[2]) ##
    list.sort()

    s = factorList[r0[1]] + ", " + factorList[r0[2]] 
    factorList.append(s)

    counter = []
    for i in range(NFACTORS):
        counter.append(1)


    for i in range(1,ITERATIONS):
         
        m1 = makeReturnSeriesMatrix1(m0, list, counter) # duplicate matrix
        r1 = getResult1(m1, list2,i, counter)
           
        list = []
        list.append(r1[1])
        list.append(r1[2])
        list2.append(r1[1])
        list2.append(r1[2])
        list.sort()
        list2.sort()

        s = factorList[r1[1]] + ", " + factorList[r1[2]] 
        factorList.append(s) 
        
        if (r1[0] < THRESHOLD): break