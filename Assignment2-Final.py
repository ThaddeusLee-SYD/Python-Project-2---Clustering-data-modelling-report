'''                       Assignment 2                                      '''


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import silhouette_score, adjusted_rand_score 
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix

#########           FEATURES OVERVIEW

staff_path = (f'G:\Python\Assignment 2\A2_HR_Employee_Data.csv')
staff_df = pd.read_csv(staff_path, sep = ',', decimal = '.', header = 0, index_col = 0)




#converting following data types from object to categories.
staff_df['BusinessTravel'] = staff_df['BusinessTravel'].astype('category')
staff_df['BusinessUnit'] = staff_df['BusinessUnit'].astype('category')
staff_df['Gender'] = staff_df['Gender'].astype('category')
staff_df['MaritalStatus'] = staff_df['MaritalStatus'].astype('category')



#Converting columns with text values to uppercase
staff_df['Resigned'] = staff_df['Resigned'].str.upper()
staff_df['BusinessTravel'] = staff_df['BusinessTravel'].str.upper()
staff_df['BusinessUnit'] = staff_df['BusinessUnit'].str.upper()
staff_df['OverTime'] = staff_df['OverTime'].str.upper()
staff_df['Gender'] = staff_df['Gender'].str.upper()
staff_df['MaritalStatus'] = staff_df['MaritalStatus'].str.upper()



#make Resigned Collumn Boolean by changing category into 1 for yes and 0 for no
staff_df['Resigned'].replace('YES', '1', inplace = True)
staff_df['Resigned'].replace('NO', '0', inplace = True)
staff_df['Resigned'] = staff_df['Resigned'].astype('int64')



#make OverTime Collumn Boolean by changing category into 1 for yes and 0 for no
staff_df['OverTime'].replace('YES', '1', inplace = True)
staff_df['OverTime'].replace('NO', '0', inplace = True)
staff_df['OverTime'] = staff_df['OverTime'].astype('int64')




staff_df = staff_df.rename(columns=str.upper)
#CMDlinetips.com
#Cmdline, 2020, How To Change Pandas Columns Names to Lower Case, cmdlinetips.com, viewed 21 Jan 2022
#<https://cmdlinetips.com/2020/07/cleaning_up_pandas-column-names/>



print(staff_df)
print(staff_df.info())








###############           DATA EXPLORATION & GRAPHS           ################


#Resigned column is Target as we are trying to determine the underlying reasons consultants are leaving. 
#firm has concerns about quality of work being produced
#higher turnover in consultants, newer consolutants not as skilled and knowledgeable.
#consultants moving to competing firms.


#YEARS IN ROLE
print(staff_df['YEARSINROLE'].value_counts())
staff_df['YEARSINROLE'].plot(kind='hist', bins=18) # bins indicates number of bins (ranges) in the histogram 

plt.title('Years in Role') # sets the title (on the "current" plot) 
plt.xlabel('Years in Role') # sets the label of the horizontal axis 
plt.ylabel('Number of Employees')
plt.grid() # turns on the grid lines
plt.show()





# YEARS AT COMPANY
print(staff_df['YEARSATCOMPANY'].value_counts())
staff_df['YEARSATCOMPANY'].plot(kind='hist', bins=40) # bins indicates number of bins (ranges) in the histogram 

plt.title('Years At Companny') # sets the title (on the "current" plot) 
plt.xlabel('Years At Companny') # sets the label of the horizontal axis 
plt.ylabel('Number of Employees')
plt.grid() # turns on the grid lines
plt.show()




## RESIGNED
#Resigned vs Business Units
Resigned_vs_Business_Units = staff_df.groupby(['BUSINESSUNIT'])['RESIGNED'].mean()
print('Resigned vs Business Units: \n', Resigned_vs_Business_Units)
Resigned_vs_Business_Units.plot(kind='bar', rot=0, title='Resigned vs Business Units') 
plt.ylabel('Percentage of Resigned')
plt.show()


# Resigned vs Monthly Income
Resigned_vs_Monthly_Income = staff_df.groupby(['RESIGNED'])['MONTHLYINCOME'].mean()
print('Resigned vs Monthly Income: \n', Resigned_vs_Monthly_Income)
#Very Informative. Mean for those that have resigned is $4779.16, compared to current employee monthly income of $6818.47
Resigned_vs_Monthly_Income.plot(kind='bar', rot=0, title='Resigned vs Monthly Income') 
plt.ylabel('Monthly Income')
plt.xlabel('Resigned \n where 0 is "Current", and 1 is "Resigned"')
plt.show()




#Resigned vs Age
AGE_vs_Resigned_mean = staff_df.groupby(['RESIGNED'])['AGE'].mean()
print('AGE vs Resigned (mean): \n', AGE_vs_Resigned_mean)

AGE_vs_Resigned_median = staff_df.groupby(['RESIGNED'])['AGE'].median()
print('AGE vs Resigned (median): \n', AGE_vs_Resigned_median)
staff_df.boxplot(column='AGE',by='RESIGNED')
plt.title('Resigned by Age')
plt.grid()
plt.ylabel('Age')
plt.xlabel('Resigned \n where 0 is "Current", and 1 is "Resigned"')
plt.show()
# Mean of those resigning is much lower. Resigned(Mean) 33.698745, Current(Mean) 37.522160
# Median - Resigned(Median) 32, Current(Median) 36.0




#Resigned vs Job Satisfaction
Resigned_vs_Job_Satisfaction_median = staff_df.groupby(['RESIGNED'])['JOBSATISFACTION'].median()
print('Resigned vs Job Satisfaction(Median): \n', Resigned_vs_Job_Satisfaction_median)
Resigned_vs_Job_Satisfaction_median.plot(kind='bar', rot=0, title='Resigned vs Job Satisfaction (Median)')
plt.xlabel('Resigned')
plt.ylabel("Job Satisfaction Level") 
plt.show()  
#### median is same across current staff and resigned.

Resigned_vs_Job_Satisfaction_mean = staff_df.groupby(['RESIGNED'])['JOBSATISFACTION'].mean()
print('Resigned vs Job Satisfaction(Mean): \n', Resigned_vs_Job_Satisfaction_mean)
Resigned_vs_Job_Satisfaction_mean.plot(kind='bar', rot=0, title='Resigned vs Job Satisfaction (Mean)')
plt.xlabel('Resigned')
plt.ylabel("Job Satisfaction Level") 
plt.show()  
### When we look at the mean Job satisfaction is lower with those that have resigned 2.468354
#compared with current employees of 2.778589







#Resigned vs Work Life Balance
Resigned_vs_Work_Life_Balance_median = staff_df.groupby(['RESIGNED'])['WORKLIFEBALANCE'].median()
print('Resigned vs Work Life Balance (median): \n', Resigned_vs_Work_Life_Balance_median)
Resigned_vs_Work_Life_Balance_median.plot(kind='bar', rot=0, title='Resigned by Work Life Balance (median)') 
plt.xlabel('Resigned \n where 0 is "Current", and 1 is "Resigned"')
plt.ylabel("Work Life Balance")
plt.show()
### Median - we see no real correlation between resigning and work life balance#

Resigned_vs_Work_Life_Balance_mean = staff_df.groupby(['RESIGNED'])['WORKLIFEBALANCE'].mean()
print('Resigned vs Work Life Balance (mean): \n', Resigned_vs_Work_Life_Balance_mean)
Resigned_vs_Work_Life_Balance_mean.plot(kind='bar', rot=0, title='Resigned by Work Life Balance (mean)') 
plt.xlabel('Resigned \n where 0 is "Current", and 1 is "Resigned"')
plt.ylabel("Work Life Balance")
plt.show()
# Mean - we see that work life balance is rated slightly lower with Resigned = 2.658228, Current = 2.781022


staff_df.boxplot(column='WORKLIFEBALANCE',by='RESIGNED')
plt.title('Resigned by Work Life Balance')
plt.grid()
plt.ylabel('Work Life Balance')
plt.xlabel('Resigned \n where 0 is "Current", and 1 is "Resigned"')
plt.show()





#Resigned vs Years in Role
Resigned_vs_Years_in_Role = staff_df.groupby(['RESIGNED'])['YEARSINROLE'].mean()
print('Resigned vs Years in Role: \n', Resigned_vs_Years_in_Role)
Resigned_vs_Years_in_Role.plot(kind='bar', rot=0, title='Resigned vs Years in Role') 
plt.xlabel('Resigned \n where 0 is "Current", and 1 is "Resigned"')
plt.ylabel("Years in Role") 
plt.show()
#  Employees who've resigned have a mean of 2.9288 years,
# Current employees have a mean of 4.4803 years



#Resigned vs Years at company
Resigned_vs_Years_at_Company = staff_df.groupby(['RESIGNED'])['YEARSATCOMPANY'].mean()
print('Resigned vs Years at Company: \n', Resigned_vs_Years_at_Company)
Resigned_vs_Years_at_Company.plot(kind='bar', rot=0, title='Resigned vs Years at Company') 
plt.xlabel('Resigned \n where 0 is "Current", and 1 is "Resigned"')
plt.ylabel("Years at Company") 
plt.show()
# RESIGNED Years at company
#0    7.369019
#1    5.130802


#Resigned vs Average Weekly hours worked
Resigned_vs_AverageWeeklyHoursWorked = staff_df.groupby(['RESIGNED'])['AVERAGEWEEKLYHOURSWORKED'].mean()
print('Resigned vs Years at Company: \n', Resigned_vs_AverageWeeklyHoursWorked)
Resigned_vs_AverageWeeklyHoursWorked.plot(kind='bar', rot=0, title='Resigned vs Average Weekly Hours') 
plt.xlabel('Resigned \n where 0 is "Current", and 1 is "Resigned"')
plt.ylabel("Average Weekly Hours") 
plt.show()
# RESIGNED
#0    42.289538
#1    46.957806

# Resignation by Marital Status
Marital_Status_to_Resigned = staff_df.groupby(['MARITALSTATUS'])['RESIGNED'].mean()
Marital_Status_to_Resigned.plot(kind='bar', rot=0, title='Resignation - By Marital Status ') 
plt.xlabel('Marital Status')
plt.ylabel("Percentage")
plt.grid()
plt.show()




# Resignation by Gender
print(staff_df.groupby(['GENDER'])['RESIGNED'].mean())
Resignation_by_Gender = staff_df.groupby(['GENDER'])['RESIGNED'].mean()
Resignation_by_Gender.plot(kind='bar', rot=0, title='Resignation - by Gender ') 
plt.xlabel('Gender')
plt.ylabel("Percentage")
plt.grid()
plt.show()
#FEMALE    0.147959
#MALE      0.170068




#Training vs Resigned
Training_vs_Resigned = staff_df.groupby(['RESIGNED'])['TRAININGTIMESLASTYEAR'].mean()
print('Training vs Resigned: \n', Training_vs_Resigned)
Training_vs_Resigned.plot(kind='bar', rot=0, title='Training Times in Last Year - Resigned ') 
plt.xlabel('Resigned \n where 0 is "Current", and 1 is "Resigned"')
plt.ylabel("Training Times in Year") 
plt.show()

#Resigned/ Trainingg:
#Current     2.832393
#Resigned    2.610879



#Resigned by Business Travel
print(staff_df.groupby(['BUSINESSTRAVEL'])['RESIGNED'].value_counts())
print(staff_df.groupby(['BUSINESSTRAVEL'])['RESIGNED'].mean())
#BUSINESSTRAVEL
#NON-TRAVEL           0.080000
#TRAVEL_FREQUENTLY    0.249097 - 24.9% more likely to resign.
#TRAVEL_RARELY        0.149569



#Performance Rating by resigned
print(staff_df.groupby(['PERFORMANCERATING'])['RESIGNED'].value_counts())
print(staff_df.groupby(['PERFORMANCERATING'])['RESIGNED'].mean())
#PERFORMANCERATING - Resigned (mean) - no strong correlation, but percentage for resigning is slightly higher for higher rated performers.
#3    0.160772
#4    0.163717




#resigned by Performance Rating
print(staff_df.groupby(['RESIGNED'])['PERFORMANCERATING'].value_counts())
print(staff_df.groupby(['RESIGNED'])['PERFORMANCERATING'].mean())
#RESIGNED - by perfomance rating
#0    3.153285
#1    3.156118






## BUSINESS UNIT
#Years Since Last Promotion by Business Unit:
Years_Since_Last_Promotion_by_Business_Unit = staff_df.groupby(['BUSINESSUNIT'])['YEARSINROLE'].mean()
print('Training Times in Last Year vs Year in Role: \n', Years_Since_Last_Promotion_by_Business_Unit)
Years_Since_Last_Promotion_by_Business_Unit.plot(kind='bar', rot=0, title='Years Since Last Promotion by Business Unit') 
plt.xlabel('Business Unit')
plt.ylabel("Years in Role") 
plt.show()
#Business Operations have shortest - 3.539683
#Consultants on average spend 4.155047 years
#Sales on average spend 4.485547 years in role






#Business unit and Job Satisfaction:
Business_unit_and_Job_Satisfaction_median = staff_df.groupby(['BUSINESSUNIT'])['JOBSATISFACTION'].median()
print('Business_unit_and_Job_Satisfaction: \n', Business_unit_and_Job_Satisfaction_median)
Business_unit_and_Job_Satisfaction_median.plot(kind='bar', rot=0, title='Business unit and Job Satisfaction (Median)') 
plt.ylabel("Mean Satisfaction Rating") 
plt.show()

Business_unit_and_Job_Satisfaction_mean = staff_df.groupby(['BUSINESSUNIT'])['JOBSATISFACTION'].mean()
print('Business_unit_and_Job_Satisfaction: \n', Business_unit_and_Job_Satisfaction_mean)
Business_unit_and_Job_Satisfaction_mean.plot(kind='bar', rot=0, title='Business unit and Job Satisfaction (Mean)') 
plt.ylabel("Mean Satisfaction Rating") 
plt.show()
# BUSINESS OPERATIONS - Satisfaction Mean Rating   2.603175
# CONSULTANTS - Satisfaction Mean Rating           2.726327
# SALES - Satisfaction Mean Rating                 2.751121



#Business unit and Work Life Balance:
Business_unit_and_Work_Life_Balance_median = staff_df.groupby(['BUSINESSUNIT'])['WORKLIFEBALANCE'].median()
print('Business unit and Work Life Balance (Median): \n', Business_unit_and_Work_Life_Balance_median)
Business_unit_and_Work_Life_Balance_median.plot(kind='bar', rot=0, title='Business unit and Work Life Balance (Median)') 
plt.ylabel("Work Life Balance") 
plt.show()

Business_unit_and_Work_Life_Balance_mean = staff_df.groupby(['BUSINESSUNIT'])['WORKLIFEBALANCE'].mean()
print('Business unit and Work Life Balance (mean): \n', Business_unit_and_Work_Life_Balance_mean)
Business_unit_and_Work_Life_Balance_mean.plot(kind='bar', rot=0, title='Business unit and Work Life Balance (Mean)') 
plt.ylabel("Work Life Balance") 
plt.show()
# Mean Average of Work Life Balance accross Business Units
#BUSINESS OPERATIONS    2.920635
#CONSULTANTS            2.725286
#SALES                  2.816143





#Business Unit vs Monthly Income:
Business_Unit_vs_Monthly_Income = staff_df.groupby(['BUSINESSUNIT'])['MONTHLYINCOME'].mean()
print('Business Unit vs Monthly Income: \n', Business_Unit_vs_Monthly_Income)
Business_Unit_vs_Monthly_Income.plot(kind='bar', rot=0, title='Monthly Income by Buisness Unit ') 
plt.xlabel('Business Unit')
plt.ylabel("Monthly Income") 
plt.show()
#Monthly Incomes per Business Unit
#BUSINESS OPERATIONS    6654.507937
#CONSULTANTS            6281.252862
#SALES                  6959.172646


Business_Unit_vs_AVERAGEWEEKLYHOURSWORKED = staff_df.groupby(['BUSINESSUNIT'])['AVERAGEWEEKLYHOURSWORKED'].mean()
print('Business Unit vs Average Weekly Hours: \n', Business_Unit_vs_AVERAGEWEEKLYHOURSWORKED)
Business_Unit_vs_AVERAGEWEEKLYHOURSWORKED.plot(kind='bar', rot=0, title='Business Unit vs Average Weekly Hours ') 
plt.xlabel('Business Unit')
plt.ylabel("Average Weekly Hours") 
plt.show()



## Performance Rating
#Business Performance vs Business Unit
Business_Performance_vs_Business_Unit = staff_df.groupby(['BUSINESSUNIT'])['PERFORMANCERATING'].median()
print('Business Performance vs Business Unit: \n', Business_Performance_vs_Business_Unit)




#Business Performance vs Years in Role
Business_Performance_vs_Years_in_Role_median = staff_df.groupby(['YEARSINROLE'])['PERFORMANCERATING'].median()
print('Business Performance vs Years in Role (median): \n', Business_Performance_vs_Years_in_Role_median)

Business_Performance_vs_Years_in_Role_mean = staff_df.groupby(['YEARSINROLE'])['PERFORMANCERATING'].mean()
print('Business Performance vs Years in Role  (mean): \n', Business_Performance_vs_Years_in_Role_mean)



staff_df.boxplot(column='PERFORMANCERATING',by='YEARSINROLE')
plt.show()






#Years in Role vs Job Satisfaction
Years_in_Role_vs_Job_Satisfaction_median = staff_df.groupby(['YEARSINROLE'])['JOBSATISFACTION'].median()
print('Years in Role vs Job Satisfaction (Median): \n', Years_in_Role_vs_Job_Satisfaction_median)
# Mean of 2.9184 times in within first year (year 0), but this drops after year 1 and  2

Years_in_Role_vs_Job_Satisfaction_median.plot(kind='bar', rot=0, title='Years in Role vs Job Satisfaction (Median)') 
plt.xlabel('Years in Role')
plt.ylabel("Job Satisfaction Level")
plt.grid()
plt.show()

Years_in_Role_vs_Job_Satisfaction_mean = staff_df.groupby(['YEARSINROLE'])['JOBSATISFACTION'].mean()
print('Years in Role vs Job Satisfaction (Mean): \n', Years_in_Role_vs_Job_Satisfaction_mean)
# Mean of 2.9184 times in within first year (year 0), but this drops after year 1 and  2

Years_in_Role_vs_Job_Satisfaction_mean.plot(kind='bar', rot=0, title='Years in Role vs Job Satisfaction (Mean)') 
plt.xlabel('Years in Role')
plt.ylabel("Job Satisfaction Level")
plt.grid()
plt.show()
# dips after 1 year in role before rising in years 4-5. suggests this is in line with promotion?? 









#Years in Role vs Work Life Balance
Years_in_Role_vs_WorkLifeBalance_median = staff_df.groupby(['YEARSINROLE'])['WORKLIFEBALANCE'].median()
print('Years in Role vs Work Life Balance (Median): \n', Years_in_Role_vs_WorkLifeBalance_median)
# Mean of 2.9184 times in within first year (year 0), but this drops after year 1 and  2

Years_in_Role_vs_WorkLifeBalance_median.plot(kind='bar', rot=0, title='Years in Role vs Work Life Balance (Median)') 
plt.xlabel('Years in Role')
plt.ylabel("Work Life Balance Level")
plt.grid()
plt.show()

Years_in_Role_vs_WorkLifeBalance_mean = staff_df.groupby(['YEARSINROLE'])['WORKLIFEBALANCE'].mean()
print('Years in Role vs Work Life Balance (Mean): \n', Years_in_Role_vs_WorkLifeBalance_mean)
# Mean of 2.9184 times in within first year (year 0), but this drops after year 1 and  2

Years_in_Role_vs_WorkLifeBalance_mean.plot(kind='bar', rot=0, title='Years in Role vs Work Life Balance (Mean)') 
plt.xlabel('Years in Role')
plt.ylabel("Work Life Balance Level")
plt.grid()
plt.show()






#Years in Role by Gender
Years_in_Role_vs_Gender_Median = staff_df.groupby(['GENDER'])['YEARSINROLE'].median()
print('Years in Role vs Gender (Median): \n', Years_in_Role_vs_Gender_Median)
#Years in Role vs Gender (Median): 
#GENDER
#FEMALE    3.0
#MALE      3.0

Years_in_Role_vs_Gender_Mean = staff_df.groupby(['GENDER'])['YEARSINROLE'].mean()
print('Years in Role vs Gender (Mean): \n', Years_in_Role_vs_Gender_Mean)
#Years in Role vs Gender (Mean): 
#GENDER
#FEMALE    4.413265
#MALE      4.106576

Years_in_Role_vs_Gender_Mean.plot(kind='bar', rot=0, title='Years in Role vs Gender (Mean)') 
plt.xlabel('Gender')
plt.ylabel("Years in Role")
plt.grid()
plt.show()







#Years Since Last Promotion vs Job Satisfaction
YearsSinceLastPromotion_vs_Job_Satisfaction_median = staff_df.groupby(['YEARSSINCELASTPROMOTION'])['JOBSATISFACTION'].median()
print('Years in Role vs Job Satisfaction (Median): \n', YearsSinceLastPromotion_vs_Job_Satisfaction_median)
# Mean of 2.9184 times in within first year (year 0), but this drops after year 1 and  2

YearsSinceLastPromotion_vs_Job_Satisfaction_median.plot(kind='bar', rot=0, title='Years Since Last Promotion vs Job Satisfaction (Median)') 
plt.xlabel('Years Since Last Promotion')
plt.ylabel("Job Satisfaction Level")
plt.grid()
plt.show()

YearsSinceLastPromotion_vs_Job_Satisfaction_mean = staff_df.groupby(['YEARSSINCELASTPROMOTION'])['JOBSATISFACTION'].mean()
print('Years in Role vs Job Satisfaction (Mean): \n', YearsSinceLastPromotion_vs_Job_Satisfaction_mean)
# Mean of 2.9184 times in within first year (year 0), but this drops after year 1 and  2

YearsSinceLastPromotion_vs_Job_Satisfaction_mean.plot(kind='bar', rot=0, title='Years Since Last Promotion vs Job Satisfaction (Mean)') 
plt.xlabel('Years Since Last Promotion')
plt.ylabel("Job Satisfaction Level")
plt.grid()
plt.show()








#Years Since Last Promotion vs Work Life Balance
YearsSinceLastPromotion_vs_WorkLifeBalance_median = staff_df.groupby(['YEARSSINCELASTPROMOTION'])['WORKLIFEBALANCE'].median()
print('Years Since Last Promotion vs Work Life Balance (Median): \n', YearsSinceLastPromotion_vs_WorkLifeBalance_median)
# Mean of 2.9184 times in within first year (year 0), but this drops after year 1 and  2

YearsSinceLastPromotion_vs_WorkLifeBalance_median.plot(kind='bar', rot=0, title='Years Since Last Promotion vs Work Life Balance (Median)') 
plt.xlabel('Years Since Last Promotion')
plt.ylabel("Work Life Balance Level")
plt.grid()
plt.show()

YearsSinceLastPromotion_vs_WorkLifeBalance_mean = staff_df.groupby(['YEARSSINCELASTPROMOTION'])['WORKLIFEBALANCE'].mean()
print('Years Since Last Promotion vs Work Life Balance (Mean): \n', YearsSinceLastPromotion_vs_WorkLifeBalance_mean)
# Mean of 2.9184 times in within first year (year 0), but this drops after year 1 and  2

YearsSinceLastPromotion_vs_WorkLifeBalance_mean.plot(kind='bar', rot=0, title='Years Since Last Promotion vs Work Life Balance (Mean)') 
plt.xlabel('Years Since Last Promotion')
plt.ylabel("Work Life Balance Level")
plt.grid()
plt.show()



#Years Since Last Promotion vs Resign
YearsSinceLastPromotion_vs_Resigned_median = staff_df.groupby(['RESIGNED'])['YEARSSINCELASTPROMOTION'].median()
print('Years Since Last Promotion by Resigned (Median): \n', YearsSinceLastPromotion_vs_Resigned_median)

YearsSinceLastPromotion_vs_Resigned_mean = staff_df.groupby(['RESIGNED'])['YEARSSINCELASTPROMOTION'].mean()
print('Years Since Last Promotion by Resigned (Median): \n', YearsSinceLastPromotion_vs_Resigned_mean)

staff_df.boxplot(column='YEARSSINCELASTPROMOTION',by='RESIGNED')
plt.title('Resigned by Years Since Last Promotion')
plt.grid()
plt.ylabel('Years Since Last Promotion')
plt.xlabel('Resigned \n where 0 is "Current", and 1 is "Resigned"')
plt.show()








#Years at Company vs Job Satisfaction
Years_at_Company_vs_Job_Satisfaction_median = staff_df.groupby(['YEARSATCOMPANY'])['JOBSATISFACTION'].median()
print('Years at Company vs Job Satisfaction (Median): \n', Years_at_Company_vs_Job_Satisfaction_median)


Years_at_Company_vs_Job_Satisfaction_median.plot(kind='bar', rot=0, title='Years at Company vs Job Satisfaction (Median)') 
plt.xlabel('Years at Company')
plt.ylabel("Job Satisfaction Level")
plt.grid()
plt.show()


Years_at_Company_vs_Job_Satisfaction_mean = staff_df.groupby(['YEARSATCOMPANY'])['JOBSATISFACTION'].mean()
print('Years in Role vs Job Satisfaction (Mean): \n', Years_at_Company_vs_Job_Satisfaction_mean)


Years_at_Company_vs_Job_Satisfaction_mean.plot(kind='bar', rot=0, title='Years at Company vs Job Satisfaction (Mean)') 
plt.xlabel('Years at Company')
plt.ylabel("Job Satisfaction Level")
plt.grid()
plt.show()








#Years at Company vs Work Life Balance
Years_at_Company_vs_WorkLifeBalance_median = staff_df.groupby(['YEARSATCOMPANY'])['WORKLIFEBALANCE'].median()
print('Years at Company vs Work Life Balance (Median): \n', Years_at_Company_vs_WorkLifeBalance_median)


Years_at_Company_vs_WorkLifeBalance_median.plot(kind='bar', rot=0, title='Years at Company vs Work Life Balance (Median)') 
plt.xlabel('Years at Company')
plt.ylabel("Work Life Balance Level")
plt.grid()
plt.show()


Years_at_Company_vs_WorkLifeBalance_mean = staff_df.groupby(['YEARSATCOMPANY'])['WORKLIFEBALANCE'].mean()
print('Years in Role vs Work Life Balance (Mean): \n', Years_at_Company_vs_WorkLifeBalance_mean)


Years_at_Company_vs_WorkLifeBalance_mean.plot(kind='bar', rot=0, title='Years at Company vs Work Life Balance (Mean)') 
plt.xlabel('Years at Company')
plt.ylabel("Work Life Balance Level")
plt.grid()
plt.show()










#Monthly Income vs Years in Role
salary_vs_years_in_role = staff_df.groupby(['YEARSINROLE'])['MONTHLYINCOME'].mean()
print('Years in Role vs Monthly Income', salary_vs_years_in_role)
staff_df.boxplot(column='MONTHLYINCOME',by='YEARSINROLE')
plt.ylabel("Monthly Income") 
plt.show()





#Years in Role and Percentage Salary Hike
Years_in_Role_and_Percentage_Salary_Hike = staff_df.groupby(['YEARSINROLE'])['PERCENTSALARYHIKE'].mean()
print('Years in Role and Percentage Salary Hike: \n', Years_in_Role_and_Percentage_Salary_Hike)

#Training Times in Last Year vs Year in Role
Training_Times_in_Last_Year_vs_Year_in_Role = staff_df.groupby(['YEARSINROLE'])['TRAININGTIMESLASTYEAR'].mean()
print('Training Times in Last Year vs Year in Role: \n', Training_Times_in_Last_Year_vs_Year_in_Role)
# Mean of 2.9184 times in within first year (year 0), but this drops after year 1 and  2







#Performance Rating vs Training Times Last Year
Performance_Rating_vs_Training_Times_Last_Year = staff_df.groupby(['PERFORMANCERATING'])['TRAININGTIMESLASTYEAR'].mean()
print('Performance Rating vs Training Times Last Year: \n', Performance_Rating_vs_Training_Times_Last_Year)
Performance_Rating_vs_Training_Times_Last_Year.plot(kind='bar', rot=0, title='Performance Rating vs Training Times Last Year ') 
plt.xlabel('Performance Rating')
plt.ylabel("Training Times in Year")
plt.grid()
plt.show()
# Mean of 2.9184 times in within first year (year 0), but this drops after year 1 and  2     
      
 #Performance Rating/ Times Trained(mean)     
#2    3.000000
#3    2.805112
#4    2.748899






# Gender vs Monthly Incomes
Gender_vs_Monthly_Income = staff_df.groupby(['GENDER'])['MONTHLYINCOME'].mean()
print('Performance Rating vs Training Times Last Year: \n', Gender_vs_Monthly_Income)
Gender_vs_Monthly_Income.plot(kind='bar', rot=0, title='Monthly Incomes - By Gender ') 
plt.xlabel('Gender')
plt.ylabel("Monthly Income")
plt.grid()
plt.show()









print(staff_df.groupby(['EDUCATIONLEVEL'])['JOBSATISFACTION'].median())
print(staff_df.groupby(['EDUCATIONLEVEL'])['JOBSATISFACTION'].mean())
####EDUCATIONLEVEL
#1    2.800000
#2    2.769504
#3    2.652098
#4    2.786432
#5    2.666667






#Resignation by performance level
print(staff_df.groupby(['RESIGNED'])['PERFORMANCERATING'].median())
print(staff_df.groupby(['RESIGNED'])['PERFORMANCERATING'].mean())
# RESIGNED by Performance Rating
#Current    3.153285
#Resigned    3.156118

Resignation_by_performance_level = staff_df.groupby(['RESIGNED'])['PERFORMANCERATING'].mean()
Resignation_by_performance_level.plot(kind='bar', rot=0, title='Resignation - By Performance Rating ') 
plt.xlabel('Resigned')
plt.ylabel("Performance Rating (mean)")
plt.grid()
plt.show()
# performance rating has no correlation to people resigning.






#Resgnation by Education Level
print(staff_df.groupby(['RESIGNED'])['EDUCATIONLEVEL'].median())
print(staff_df.groupby(['RESIGNED'])['EDUCATIONLEVEL'].mean())
#RESIGNED By Education Level (mean)
#0    2.927007
#1    2.839662
#slightly lower but no significant correlation



#Job Satisfaction and Business Travel
print(staff_df.groupby(['BUSINESSTRAVEL'])['JOBSATISFACTION'].median())
print(staff_df.groupby(['BUSINESSTRAVEL'])['JOBSATISFACTION'].mean())
#Mean -
#NON-TRAVEL           2.793333
#TRAVEL_FREQUENTLY    2.790614
#TRAVEL_RARELY        2.702780


#Work Life Balance and Business Travel
print(staff_df.groupby(['BUSINESSTRAVEL'])['WORKLIFEBALANCE'].median())
print(staff_df.groupby(['BUSINESSTRAVEL'])['WORKLIFEBALANCE'].mean())
#Mean below is slightly lower for those who travel rarely, but not real correlation
#NON-TRAVEL           2.773333
#TRAVEL_FREQUENTLY    2.776173
#TRAVEL_RARELY        2.755513



#Job Satisfaction and Overtime.
print(staff_df.groupby(['BUSINESSTRAVEL'])['OVERTIME'].value_counts())
print(staff_df.groupby(['BUSINESSTRAVEL'])['OVERTIME'].mean())
#BUSINESSTRAVEL and Overtime
#NON-TRAVEL           0.233333
#TRAVEL_FREQUENTLY    0.310469 - Employees that tend to travel either frequently or rarely do more overtime
#TRAVEL_RARELY        0.282838


print(staff_df.groupby(['BUSINESSUNIT'])['OVERTIME'].value_counts())
print(staff_df.groupby(['BUSINESSUNIT'])['OVERTIME'].mean())
#BUSINESSUNIT - Overtime
#BUSINESS OPERATIONS    0.269841
#CONSULTANTS            0.281998
#SALES                  0.286996
# Percentage of Sales, followed by consultants tend to do most overtime



print(staff_df.groupby(['BUSINESSTRAVEL'])['AVERAGEWEEKLYHOURSWORKED'].mean())
#BUSINESSTRAVEL
#NON-TRAVEL           42.240000
#TRAVEL_FREQUENTLY    43.357401
#TRAVEL_RARELY        43.073826
#Average hours of those travelling frequently is just over an hour more on average than those that don't travel.




#Gender vs Years in Role
print(staff_df.groupby(['GENDER'])['YEARSINROLE'].mean())
#GENDER - As we can see below, Men are more likely to resign in a shorter period.
#FEMALE    4.413265
#MALE      4.106576

# Job Satisfaction by Over Time
print(staff_df.groupby(['OVERTIME'])['JOBSATISFACTION'].median())
print(staff_df.groupby(['OVERTIME'])['JOBSATISFACTION'].mean())
# Job Satisfaction by Overtime
#0    2.711575
#1    2.771635
#Overtime has no correlation to job satisfaction

# Job Satisfaction by Work Life Balance
print(staff_df.groupby(['OVERTIME'])['WORKLIFEBALANCE'].median())
print(staff_df.groupby(['OVERTIME'])['WORKLIFEBALANCE'].mean())
# on average, Work Life Balance for those doing overtime is slightly lower.
#0    2.773245
#1    2.730769
# no significant correlation





#Marital Status - Monthly Income
print(staff_df.groupby(['MARITALSTATUS'])['MONTHLYINCOME'].mean())



#Marital Status - Job Satisfaction
print(staff_df.groupby(['MARITALSTATUS'])['JOBSATISFACTION'].median())
print(staff_df.groupby(['MARITALSTATUS'])['JOBSATISFACTION'].mean())
#MARITALSTATUS and Job Satisfaction Mean- mean is lowest among divorced, but higher percentage of single resigning
#DIVORCED    2.697248
#MARRIED     2.716196
#SINGLE      2.768085





#Marital Status - Work Life Balance
print(staff_df.groupby(['MARITALSTATUS'])['WORKLIFEBALANCE'].median())
print(staff_df.groupby(['MARITALSTATUS'])['WORKLIFEBALANCE'].mean())
#MARITALSTATUS (mean average)
#DIVORCED    2.749235
#MARRIED     2.756315
#SINGLE      2.776596


#Job Satisfaction vs Number of Companies worked
print(staff_df.groupby(['JOBSATISFACTION'])['NUMCOMPANIESWORKED'].median())
#JOBSATISFACTION - median
#1    2.0
#2    2.0
#3    2.0
#4    1.0

print(staff_df.groupby(['JOBSATISFACTION'])['NUMCOMPANIESWORKED'].mean())
#1    2.871972
#2    2.839286
#3    2.667421
#4    2.516340


#Resigned by Average weekly hours
print(staff_df.groupby(['RESIGNED'])['AVERAGEWEEKLYHOURSWORKED'].mean())
#RESIGNED
#0    42.289538
#1    46.957806

print(staff_df.groupby(['OVERTIME'])['RESIGNED'].mean())
#OVERTIME people who did Overtime have a higher percentage of resigning
#0    0.104364
#1    0.305288


# Resigned by Number of Companies Worked
print(staff_df.groupby(['RESIGNED'])['NUMCOMPANIESWORKED'].median())
print(staff_df.groupby(['RESIGNED'])['NUMCOMPANIESWORKED'].mean())
#RESIGNED mean - number of companies
#0    2.645580
#1    2.940928


#Marital Status by age
print(staff_df.groupby(['MARITALSTATUS'])['AGE'].mean())

#resigned - percentage salary hike
print(staff_df.groupby(['RESIGNED'])['PERCENTSALARYHIKE'].mean())




print(staff_df.groupby(['YEARSINROLE'])['PERCENTSALARYHIKE'].mean())


staff_by_gender = staff_df.groupby(['GENDER'])['AGE'].value_counts()
print(staff_by_gender)
print(staff_df.groupby(['GENDER'])['AGE'].mean())
#GENDER
#FEMALE    37.329932
#MALE      36.653061



#Gender Years at Company
print(staff_df.groupby(['GENDER'])['YEARSATCOMPANY'].mean())
#GENDER - Years at company
#FEMALE    7.231293
#MALE      6.859410


#Business Unit By Age
print(staff_df.groupby(['BUSINESSUNIT'])['AGE'].mean())








print(staff_df.groupby(['RESIGNED','GENDER', 'BUSINESSUNIT'])['MONTHLYINCOME'].mean()) 

'''RESIGNED  GENDER  BUSINESSUNIT       
0         FEMALE  BUSINESS OPERATIONS    8761.142857
                  CONSULTANTS            6808.166667
                  SALES                  7328.039735
#         MALE    BUSINESS OPERATIONS    6810.513514
#                 CONSULTANTS            6508.873984
                  SALES                  7160.980296
1         FEMALE  BUSINESS OPERATIONS    3770.666667
                  CONSULTANTS            4212.6744193
                  SALES                  5557.842105
          MALE    BUSINESS OPERATIONS    3660.833333
                  CONSULTANTS            4058.100000
                  SALES                  6155.185185
'''


NewStaff = staff_df['YEARSATCOMPANY'] <= 2
print(NewStaff)

WorkLifeBalance_Young_mean = staff_df.loc[NewStaff, 'WORKLIFEBALANCE'].mean()
print(WorkLifeBalance_Young_mean)

WorkLifeBalance_Young_median = staff_df.loc[NewStaff, 'WORKLIFEBALANCE'].median()
print(WorkLifeBalance_Young_median)

NewStaff_Age = staff_df.loc[NewStaff, 'AGE'].mean()
print(NewStaff_Age)



#Job Satisfaction by Monthly Income
print(staff_df.groupby(['JOBSATISFACTION'])['MONTHLYINCOME'].median())
#JOBSATISFACTION
#1    4968.0
#2    4853.0
#3    4788.5
#4    5126.0


print(staff_df.groupby(['JOBSATISFACTION'])['MONTHLYINCOME'].mean())
#JOBSATISFACTION
#1    6561.570934
#2    6527.328571
#3    6480.495475
#4    6472.732026






#Work Life Balance - Monthly Income
print(staff_df.groupby(['WORKLIFEBALANCE'])['MONTHLYINCOME'].median())
#WORKLIFEBALANCE
#1    4269.5
#2    4970.0
#3    4941.0
#4    5067.0



print(staff_df.groupby(['WORKLIFEBALANCE'])['MONTHLYINCOME'].mean())
#WORKLIFEBALANCE
#1    5887.137500
#2    6461.808140
#3    6532.232923
#4    6746.352941










print(staff_df.groupby(['BUSINESSUNIT'])['GENDER'].value_counts())
# We see that company has more Men in workforce than women

#Gender - Job Satisfaction
print(staff_df.groupby(['GENDER'])['JOBSATISFACTION'].mean())
#GENDER - Job Satisfaction (mean)
#FEMALE    2.683673
#MALE      2.758503

print(staff_df.groupby(['GENDER'])['JOBSATISFACTION'].median())
#GENDER - Job Satisfaction (median)
#FEMALE    3.0
#MALE      3.0



#Gender - Work Life Balance
print(staff_df.groupby(['GENDER'])['WORKLIFEBALANCE'].mean())
#GENDER - Work Life Balance (mean)
#FEMALE    2.763605
#MALE      2.759637


print(staff_df.groupby(['GENDER'])['WORKLIFEBALANCE'].median())
#GENDER - Work Life Balance (median)
#FEMALE    3.0
#MALE      3.0




#resigned by Performance Rating
print(staff_df.groupby(['RESIGNED'])['PERFORMANCERATING'].value_counts())
print(staff_df.groupby(['RESIGNED'])['PERFORMANCERATING'].mean())
#RESIGNED - by perfomance rating
#0    3.153285
#1    3.156118

#resigned by Performance Rating
print(staff_df.groupby(['NUMCOMPANIESWORKED'])['AGE'].value_counts())
print(staff_df.groupby(['NUMCOMPANIESWORKED'])['AGE'].mean())
#NUMCOMPANIESWORKED
#0    34.208122
#1    32.460653
#2    41.164384
#3    41.213836
#4    41.597122
#5    38.809524
#6    39.000000
#7    40.878378
#8    41.040816
#9    39.826923



print(staff_df.groupby(['YEARSATCOMPANY'])['OVERTIME'].value_counts())
print(staff_df.groupby(['YEARSATCOMPANY'])['OVERTIME'].mean())
#YEARSATCOMPANY
#0     0.318182
#1     0.321637
#2     0.275591
#3     0.312500
#4     0.272727
#5     0.290816
#6     0.315789
#7     0.233333
#8     0.300000
#9     0.231707
#10    0.266667
#11    0.250000
#12    0.214286
#13    0.166667
#14    0.000000
#15    0.350000
#16    0.500000
#17    0.222222
#18    0.230769
#19    0.363636
#20    0.222222
#21    0.357143
#22    0.266667
#23    0.000000
#24    0.166667
#25    0.250000
#26    0.250000
#27    0.500000
#29    0.500000
#30    1.000000
#31    0.666667
#32    0.333333
#33    0.600000
#34    0.000000
#36    0.500000
#37    0.000000
#40    0.000000



print(staff_df.groupby(['RESIGNED'])['NUMCOMPANIESWORKED'].value_counts())
print(staff_df.groupby(['RESIGNED'])['NUMCOMPANIESWORKED'].mean())
#RESIGNED
#0    2.645580
#1    2.940928



print(staff_df.groupby(['NUMCOMPANIESWORKED'])['RESIGNED'].value_counts())
print(staff_df.groupby(['NUMCOMPANIESWORKED'])['RESIGNED'].mean())
#NUMCOMPANIESWORKED
#0    0.116751
#1    0.188100
#2    0.109589
#3    0.100629
#4    0.122302
#5    0.253968
#6    0.228571
#7    0.229730
#8    0.122449
#9    0.230769

print(staff_df.groupby(['NUMCOMPANIESWORKED'])['YEARSATCOMPANY'].value_counts())


print(staff_df.groupby(['NUMCOMPANIESWORKED'])['YEARSATCOMPANY'].mean())
#NUMCOMPANIESWORKED
#0    8.253807
#1    8.030710
#2    5.808219
#3    5.886792
#4    5.647482
#5    6.000000
#6    5.371429
#7    7.445946
#8    6.326531
#9    5.923077



print(staff_df.groupby(['EDUCATIONLEVEL'])['RESIGNED'].mean())
#EDUCATIONLEVEL
#1    0.182353
#2    0.156028
#3    0.173077
#4    0.145729
#5    0.104167


EducationlevelResigned_median = staff_df.groupby(['RESIGNED'])['EDUCATIONLEVEL'].median()
EducationlevelResigned_median.plot(kind='bar', rot=0, title='Resignation - Education Level ') 
plt.xlabel('Education Level')
plt.ylabel("Resigned (median)")
plt.grid()
plt.show()
#Median level is same 3

EducationlevelResigned = staff_df.groupby(['EDUCATIONLEVEL'])['RESIGNED'].mean()
EducationlevelResigned.plot(kind='bar', rot=0, title='Resignation - Education Level ') 
plt.xlabel('Education Level')
plt.ylabel("Resigned (mean)")
plt.grid()
plt.show()



print(staff_df.groupby(['RESIGNED'])['TOTALWORKINGYEARS'].mean())
print(staff_df.groupby(['RESIGNED'])['YEARSATCOMPANY'].mean())
print(staff_df.groupby(['RESIGNED'])['YEARSINROLE'].mean())

print(staff_df.groupby(['TRAININGTIMESLASTYEAR'])['TOTALWORKINGYEARS'].mean())
print(staff_df.groupby(['TRAININGTIMESLASTYEAR'])['YEARSATCOMPANY'].mean())
print(staff_df.groupby(['TRAININGTIMESLASTYEAR'])['YEARSINROLE'].mean())

sns.pairplot(staff_df, diag_kind="hist", hue='RESIGNED') 


##############################################################################

#Age vs Resigned - mean of those resigning is much lower than those who are current employees
#Resigned vs Monthly Income - mean of those resigning is much lower
#Resigned vs Years in Role:
#Resigned have lower mean of 2.9289 years.
#Current employees have mean of 4.4803 years.









'''                       Assignment 2 - Continued                   '''





staff_path = (f'G:\Python\Assignment 2\A2_HR_Employee_Data.csv')
staff_df = pd.read_csv(staff_path, sep = ',', decimal = '.', header = 0, index_col = 0)




#converting following data types from object to categories.
staff_df['BusinessTravel'] = staff_df['BusinessTravel'].astype('category')
staff_df['BusinessUnit'] = staff_df['BusinessUnit'].astype('category')
staff_df['Gender'] = staff_df['Gender'].astype('category')
staff_df['MaritalStatus'] = staff_df['MaritalStatus'].astype('category')



#Converting columns with text values to uppercase
staff_df['Resigned'] = staff_df['Resigned'].str.upper()
staff_df['BusinessTravel'] = staff_df['BusinessTravel'].str.upper()
staff_df['BusinessUnit'] = staff_df['BusinessUnit'].str.upper()
staff_df['OverTime'] = staff_df['OverTime'].str.upper()
staff_df['Gender'] = staff_df['Gender'].str.upper()
staff_df['MaritalStatus'] = staff_df['MaritalStatus'].str.upper()



#make Resigned Column Boolean by changing category into 1 for yes and 0 for no
staff_df['Resigned'].replace('YES', '1', inplace = True)
staff_df['Resigned'].replace('NO', '0', inplace = True)
staff_df['Resigned'] = staff_df['Resigned'].astype('int64')






#make OverTime Column Boolean by changing category into 1 for yes and 0 for no
staff_df['OverTime'].replace('YES', '1', inplace = True)
staff_df['OverTime'].replace('NO', '0', inplace = True)
staff_df['OverTime'] = staff_df['OverTime'].astype('int64')




staff_df = staff_df.rename(columns=str.upper)
#CMDlinetips.com
#Cmdline, 2020, How To Change Pandas Columns Names to Lower Case, cmdlinetips.com, viewed 21 Jan 2022
#<https://cmdlinetips.com/2020/07/cleaning_up_pandas-column-names/>



# Normalise column for MONTHLYINCOME
# Reference https://www.statology.org/normalize-data-in-python/

x = staff_df['MONTHLYINCOME']
staff_df['MONTHLYINCOME'] = (x - x.min())/ (x.max() - x.min())




print(staff_df)
print(staff_df.info())




#Creating Hard Copy of dataframe so that we don't make any unnecessary changes to original
staff_df2 = staff_df.copy()
staff_df2


#######           DATA MODELLING                 #####


''' We'll create the data set for running K Means and DBSCAN clustering models. We will drop the following columns
to reduce noise caused by redundant/ categorical data:
    -BUSINESSTRAVEL
    -BUSINESSUNIT
    -EDUCATIONLEVEL
    -GENDER
    -MARITALSTATUS
    -NUMCOMPANIESWORKED
    -OVERTIME
    -PERCENTSALARYHIKE
    -PERFORMANCERATING
    -TRAININGTIMESLASTYEAR
    -YEARSINROLE
    -YEARSSINCELASTPROMOTION
    -YEARSWITHCURRMANAGER

Therefore keeping the following features with our project goal in mind:
    -AGE
    -JOBSATISFACTION
    -MONTHLYINCOME
    -AVERAGEWEEKLYHOURSWORKED
    -TOTALWORKINGYEARS
    -WORKLIFEBALANCE
    -YEARSATCOMPANY
    
    
    
    
'''
#https://stackoverflow.com/questions/58697757/how-to-choose-data-columns-and-target-columns-in-a-dataframe-for-test-train-spli
staff_data_set = staff_df2.drop(columns = ['BUSINESSTRAVEL', 'BUSINESSUNIT', 'EDUCATIONLEVEL', 'GENDER', 'MARITALSTATUS', 'NUMCOMPANIESWORKED', 'OVERTIME', 'PERCENTSALARYHIKE', 'PERFORMANCERATING', 'TRAININGTIMESLASTYEAR', 'YEARSINROLE', 'YEARSWITHCURRMANAGER', 'YEARSSINCELASTPROMOTION'])
X = staff_data_set.drop(columns = ['RESIGNED'], axis = 1)
y = staff_data_set['RESIGNED']




############                    KMeans                      ##############


def k_means_staff(data_set, k):
    # replace n_clusters with the parameter k
    km_model = KMeans(n_clusters=k, random_state=14)
    km_model.fit(staff_data_set.drop(columns=['RESIGNED'])) # fit the model on the data, without the labels
    clusters = km_model.predict(staff_data_set.drop(columns=['RESIGNED'])) # assign every sample a cluster
    # replace the data_df with the iris_data_df parameter
    km_df = staff_data_set.copy()
    # add a new column to the df and assign it the clustes labels
    km_df['CLUSTERS'] = clusters
    # using loc to access all rows and 2 columns in the df, then applying value_counts and sorting by the index columns
#    print('CLUSTER:\n', km_df['CLUSTER'].value_counts())
#    print('Resigned:\n', km_df['RESIGNED'].value_counts())
    print(f'Value counts for Clusters in k = {k} are:\n', km_df['CLUSTERS'].value_counts())
    print(f'Value counts for Resigned in k = {k} are:\n', km_df['RESIGNED'].value_counts())
#    print(km_df.loc[:,['CLUSTER','RESIGNED']].value_counts().sort_index())
    
    
    
    
    
for k in (2,4,6):
   k_means_staff(staff_data_set, k)
  
   
  
    
### KMeans with selected 
km_model = KMeans(n_clusters=2, random_state=14)
km_model.fit(staff_data_set.drop(columns=['RESIGNED'])) # fit the model on the data, without the labels
clusters = km_model.predict(staff_data_set.drop(columns=['RESIGNED'])) # assign every sample a cluster
# replace the data_df with the iris_data_df parameter
km_df = staff_data_set.copy()
# add a new column to the df and assign it the clustes labels
km_df['CLUSTERS'] = clusters
# using loc to access all rows and 2 columns in the df, then applying value_counts and sorting by the index columns
#    print('CLUSTER:\n', km_df['CLUSTER'].value_counts())
#    print('Resigned:\n', km_df['RESIGNED'].value_counts())
print(f'Value counts for Clusters in k = {k} are:\n', km_df['CLUSTERS'].value_counts())
print(f'Value counts for Resigned in k = {k} are:\n', km_df['RESIGNED'].value_counts())

######## CONFUSION MATRIX for KMEANS   ################

#y_true = km_df['RESIGNED']
#y_pred = km_df['CLUSTERS']
#confusion_matrix_KM = confusion_matrix(y_true, y_pred, labels = [1,0])


#print(' The Confusion matrix for KMeans model\nTP,FN\nFP,TN\n', confusion_matrix_KM)   

#Output for Confusion Matrix using KMeans with k = 2 is:
#TP,FN
#FP,TN
# [[ 44 193]
# [386 847]]


##### Intrinsic Evaluation using Elbow Graph and Silhouette Graph to find suitable values for k

def plot_elbow_graph(data_set, k_range=range(1, 11)): 

    # A list holds the inertia values for each k 

    inertia_lst = [] 

    for k in k_range: 

        km_model = KMeans(n_clusters=k, random_state=42) 

        km_model.fit(data_set) 

        inertia_lst.append(km_model.inertia_) 

    plt.figure(figsize=(10,6)) 

    plt.plot(k_range, inertia_lst) 
    
    plt.title('WCSS Intrinsic Evaluation')

    plt.xticks(k_range) 

    plt.xlabel("Number of Clusters") 

    plt.ylabel("Inertia") 

    plt.show()
    
  
    
  

def plot_silhouette_graph(dataset, k_range=range(2, 11)): 

    # A list holds the silhouette coefficients for each k 

    silhouette_coefficients = [] 

    for k in k_range: 

        km_model = KMeans(n_clusters=k, random_state=42) 

        km_model.fit(dataset) 

        score = silhouette_score(dataset, km_model.labels_) 

        silhouette_coefficients.append(score) 

    plt.figure(figsize=(10,6)) 

    plt.plot(k_range, silhouette_coefficients) 
   
    plt.title('Silhouette Coeficcient')

    plt.xticks(k_range) 

    plt.xlabel("Number of Clusters") 

    plt.ylabel("Silhouette Coefficient") 

    plt.show()


plot_elbow_graph(X)

plot_silhouette_graph(X)







#Extrinsic Valuation using Adjusted Rand Index
######      ARI displays a max result for k= 6



ari_scores = [] 

k_range=range(1, 16) 

for k in k_range: 

    km_model = KMeans(n_clusters=k, random_state=42) 

    km_model.fit(X) 

    score = adjusted_rand_score(y, km_model.labels_) 

    ari_scores.append(score) 
    
    
    
    
    plt.figure(figsize=(10,6)) 

plt.plot(k_range, ari_scores) 

plt.xticks(k_range) 

plt.title('Adjusted Rand Index - KMeans')

plt.xlabel("Number of Clusters") 

plt.ylabel("ARI Score") 

plt.show()



staff_data_set






### This is becuase ARI gave us a result of k =6
km_model = KMeans(n_clusters=6, random_state=42) 
km_model.fit(X) 
print('ARI for k = 6 is: ', adjusted_rand_score(y, km_model.labels_)) 
# 0.029589596311280622



#############                   DBSCAN                   #############

staff_data_set2 = staff_df2.drop(columns = ['BUSINESSTRAVEL', 'BUSINESSUNIT', 'EDUCATIONLEVEL', 'GENDER', 'MARITALSTATUS', 'NUMCOMPANIESWORKED', 'OVERTIME', 'PERCENTSALARYHIKE', 'PERFORMANCERATING', 'TRAININGTIMESLASTYEAR', 'YEARSINROLE', 'YEARSWITHCURRMANAGER', 'YEARSSINCELASTPROMOTION'])
X = staff_data_set2.drop(columns = ['RESIGNED'], axis = 1)
y = staff_data_set2['RESIGNED']





#############              NEAREST NEIGHBOURS               #############
#Determine right value for Eps
'''As a rule of thumb, a minimum minPts can be derived from the number of dimensions D in the data set, as minPts ≥ D + 1. In this case. 8''' 

nbrs = NearestNeighbors(n_neighbors=20)
nbrs.fit(staff_data_set2)
distances, indices = nbrs.kneighbors(staff_data_set2)
print(f'The shape of indices array: {indices.shape}')
print(f'The shape of distances array: {distances.shape}')

# print the indices of the 5 nearest neighbours for the first sample 
print(f'The indices of the 5 nearest neighbours for the first sample: \n{indices[0][:5]}\n')
# print the distances of the 5 nearest neighbours for the first sample  
print(f'The distances of the 5 nearest neighbours for the first sample: \n{distances[0,:5]}\n')

''' Plotting K Distance Graph '''
# the index of the 5th element is 4
k_distance = distances[:,  13] # the distance of every 5th neighbour
k_distance = np.sort(k_distance) # sort in ascending order

plt.figure(figsize=(15,10)) # create and define the figure
plt.plot(k_distance)
plt.title("K-distance graph for Staff Data Set MinPts = 14")
plt.ylabel("distance")
plt.xlabel("number of core points") 
plt.gca().invert_xaxis() # gca used to get current axes, then invert_xaxis() will invert the x axis
plt.grid() # add grid to the plot
plt.show()


'''From The K Distance Graph we can see a change in the slope where the distance is equal to 5'''



###########                 DBSCAN MODELLING               ############
#reference https://stats.stackexchange.com/questions/88872/a-routine-to-choose-eps-and-minpts-for-dbscan
# Given the number of variables we chose number of features which is 7 and multiplied by 2, giving 14 for min samples.
def DBSCAN_staff(dataset, e):
    eps = e
    dbs_model = DBSCAN(eps=e, min_samples=14) # create a DBSCAN model
    dbs_model.fit(dataset.drop(columns = ['RESIGNED'])) # fit the model on the data without the labels
    clusters = dbs_model.labels_ # return a cluster for every sample from the data
    DBSCAN_df = dataset.copy()
  # add a new column to the df and assign it the clustes labels
    DBSCAN_df['CLUSTERS'] = clusters
    print(f' Clusters for eps = {e} are for: ', clusters)
    print(f'Value counts for Clusters in e = {e} :\n', DBSCAN_df['CLUSTERS'].value_counts())
    print(f'Value counts for Resigned in e = {e} :\n', DBSCAN_df['RESIGNED'].value_counts())
    
#for k in (2, 4, 6):
#    DBSCAN_staff(staff_data_set2, 0.7, k)
    
for e in (7.5, 8.25, 9):
    DBSCAN_staff(staff_data_set2, e)
    
    


#### DBSCAN VALUATION   
##### Silhouette Scores  ###########
### DBSCAN eps = 7.5, MinPts =14
dbs_model = DBSCAN(eps=7.5, min_samples=14) # create a DBSCAN model
dbs_model.fit(staff_data_set2.drop(columns = ['RESIGNED'])) # fit the model on the data without the labels
clusters = dbs_model.labels_ # return a cluster for every sample from the data
DBSCAN_df = staff_data_set2.copy()
# add a new column to the df and assign it the clustes labels
DBSCAN_df['CLUSTERS'] = clusters
print('DBSCAN Silhouette Scores eps = 7.50, MinPts =14: ', silhouette_score(DBSCAN_df, DBSCAN_df['CLUSTERS']))
# 0.4566111184563737




### DBSCAN eps = 8.25, MinPts =14
dbs_model = DBSCAN(eps=8.25, min_samples=14) # create a DBSCAN model
dbs_model.fit(staff_data_set2.drop(columns = ['RESIGNED'])) # fit the model on the data without the labels
clusters = dbs_model.labels_ # return a cluster for every sample from the data
DBSCAN_df = staff_data_set2.copy()
# add a new column to the df and assign it the clustes labels
DBSCAN_df['CLUSTERS'] = clusters
print('DBSCAN Silhouette Scores eps = 8.25, MinPts =14: ', silhouette_score(DBSCAN_df, DBSCAN_df['CLUSTERS']))
#0.43396087944129746




### DBSCAN eps = 9, MinPts =14
dbs_model = DBSCAN(eps=9, min_samples=14) # create a DBSCAN model
dbs_model.fit(staff_data_set2.drop(columns = ['RESIGNED'])) # fit the model on the data without the labels
clusters = dbs_model.labels_ # return a cluster for every sample from the data
DBSCAN_df = staff_data_set2.copy()
# add a new column to the df and assign it the clustes labels
DBSCAN_df['CLUSTERS'] = clusters
print('DBSCAN Silhouette Scores eps = 9, MinPts =14: ', silhouette_score(DBSCAN_df, DBSCAN_df['CLUSTERS']))
#0.4345867733022825




# Confusion Matrix output for DBSCAN model
#y_true = DBSCAN_df['RESIGNED']
#y_pred = DBSCAN_df['CLUSTERS']
#confusion_matrix_DBS = confusion_matrix(y_true, y_pred, labels = [1,0])\

# The Confusion matrix for DBSCAN model with eps = 8.25 and min samples 14 is:
#TP,FN
#FP,TN
#[[   1  224]
#[  11 1208]]
#print(' The Confusion matrix for DBSCAN model\nTP,FN\nFP,TN\n', confusion_matrix_DBS)    

