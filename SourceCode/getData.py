import os
import pandas as pd
import storedprocs as sp
import pyodbc
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
            query = sp.getFactorData2(2016,m,59)
            df = pd.read_sql_query(query, engine)

            if (m==1): fname = "JanFeb"
            if (m==2): fname = "FebMar"
            if (m==3): fname = "MarApr"

            filename = r"C:\\Users\\michael\\"+fname+"2016.data"

            #fill nans with mean of column
            df2 = df.fillna(df.mean())
            df2.to_csv(filename, index=False)
            #print(df2)

if __name__ == "__main__":

    makeCombinedFiles()
