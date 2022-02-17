from flask import Flask, render_template, request
import requests
import pickle
import numpy as np

from werkzeug.utils import secure_filename
import os

import pandas as pd
#import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

# Import label encoder
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")

#creating upload folder for files
UPLOAD_FOLDER = 'static/uploads/'


app = Flask(__name__, template_folder = 'templates')

#loading the pickle file
model = pickle.load(open('hr_model_new4.sav', 'rb'))

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#used to clean the folder after restarting the link
def cleandir(directo):
    for i in os.listdir(f'{directo}'):
        print(i)
        os.remove(f'{directo}/{i}')



@app.route('/',methods=['GET'])
def Home():
    return render_template('index7.html')


#Functio to predict the output
@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':

        
        #fetching the input from the form
        Age = float(request.form['Age'])
        MonthlyIncome = float(request.form['MonthlyIncome'])
        TotalWorkingYears = float(request.form['TotalWorkingYears'])
        DistanceFromHome = float(request.form['DistanceFromHome'])

        PercentSalaryHike = float(request.form['PercentSalaryHike'])
        JobRole = float(request.form['JobRole'])
        NumCompaniesWorked = float(request.form['NumCompaniesWorked'])

        City = float(request.form['City'])
        Business_type = float(request.form['Business_type'])

        YearsSinceLastPromotion = float(request.form['YearsSinceLastPromotion'])
        Education = float(request.form['Education'])
        MaritalStatus = float(request.form['MaritalStatus'])
        JobLevel = float(request.form['JobLevel'])
 
        EducationField = float(request.form['EducationField'])
        BusinessTravel = float(request.form['BusinessTravel'])
        Department = float(request.form['Department'])
        Gender = float(request.form['Gender'])

        pred = model.predict(np.array([[Age, MonthlyIncome, TotalWorkingYears, DistanceFromHome, 
                                        PercentSalaryHike, JobRole, NumCompaniesWorked,
                                        YearsSinceLastPromotion, Education, MaritalStatus, 
                                        JobLevel, EducationField, BusinessTravel, Department, Gender, Business_type, City]]))

        result = []
        
        
        #Results bsaed on condition
        if MonthlyIncome >=10000 and MonthlyIncome <50000:
            a = "The Percentage of Employee Attrition is Very High Based on Below parameters"
            i = "The Salary is Very Low"
            
            #heading.append(a)
            result.append(i)
            if MaritalStatus == 2:
                b = "The Employee Marital Status is Single"
                result.append(b)
            if DistanceFromHome >=0 and DistanceFromHome <=8:
                c = "The Distance from office is not so Far"
                result.append(c)
            if (Age >=34 and Age <=40):
                d = "The Employee Age is in middle aged of around 25 to 40"
                result.append(d)
            if TotalWorkingYears >=1 and TotalWorkingYears <=10:
                e = "The Working Years of Employee is around 2 to 10"
                result.append(e)
            if PercentSalaryHike >=11 and PercentSalaryHike <=15:    
                f = "The Hike percentage of Employee is Less"
                result.append(f)
            if City == 2:
                j = "The Employee Office location is Mohali"
                result.append(j)
                
        if MonthlyIncome >=50001 and MonthlyIncome <100000:
            a = "The Percentage of Employee Attrition is High Based on Below parameters"
            i = "The Salary is Average"
            
            #heading.append(a) 
            result.append(i)
            if MaritalStatus == 1:
                b = "The Employee Marital Status is Married"
                result.append(b)
            if DistanceFromHome >=8 and DistanceFromHome <=15:
                c = "The Distance from office is not so Far"
                result.append(c)
            if (Age >=29 and Age <=33) or (Age >=41 and Age <=47):
                d = "The Employee Age is around 30 or 45"
                result.append(d)
            if TotalWorkingYears >=2 and TotalWorkingYears <=9:
                e = "The Working Years of Employee is around 3 to 7"
                result.append(e)
            if PercentSalaryHike >=16 and PercentSalaryHike <=19:    
                f = "The Hike percentage of Employee is Satisfactory"
                result.append(f)
            if City == 1:
                j = "The Employee Office location is Delhi"
                result.append(j)

        if MonthlyIncome >=100000 and MonthlyIncome <=150000:
            a = "The Percentage of Employee Attrition is Low Based on Below parameters"
            i = "The Salary is Above Average"
            
            #heading.append(a) 
            result.append(i)
            if MaritalStatus == 0:
                b = "The Employee Marital Status is Divorced"
                result.append(b)
            if DistanceFromHome >=15 and DistanceFromHome <=20:
                c = "The Distance from office is Far"
                result.append(c)
            if (Age >=24 and Age <=28) or (Age >=48 and Age <=55):
                d = "The Employee Age is around '24 to 28 or 28 to 55'"
                result.append(d)
            if TotalWorkingYears >=13 and TotalWorkingYears <=24:
                e = "The Working Years of Employee is around 12 to 24"
                result.append(e)
            if PercentSalaryHike >=20 and PercentSalaryHike <=22:    
                f = "The Hike percentage of Employee is Good"
                result.append(f)
            if City == 3:
                j = "The Employee Office location is Pune"
                result.append(j)
                
        if MonthlyIncome >=150001 and MonthlyIncome <=300000:
            a = "The Percentage of Employee Attrition is Very Low Based on Below parameters"
            i = "The Salary is High"
            
            #heading.append(a)
            result.append(i)
            if MaritalStatus == 0:
                b = "The Employee Marital Status is Divorced"
                result.append(b)
            if DistanceFromHome >=20 and DistanceFromHome <=30:
                c= "The Distance from office is Very Far"
                result.append(c)
            if (Age >=18 and Age <=23) or (Age >=56 and Age <=60):
                d = "The Employee Age is young or in years around Retirement"
                result.append(d)
            if TotalWorkingYears >=25 and TotalWorkingYears <=40:
                e = "The Working Years of Employee is more"
                result.append(e)
            if PercentSalaryHike >=23 and PercentSalaryHike <=25:    
                f = "The Hike percentage of Employee is as Expected"
                result.append(f)
            if City == 0:
                j = "The Employee Office location is Bangalore"
                result.append(j)

        print(pred)

        #print(model.feature_importances_)

        if pred == 0 and MonthlyIncome <100000:
            return render_template('index7.html', feature = result, heading = a )
        else:
            return render_template('index7.html', feature = result, heading = a )
    else:
        return render_template('index7.html')

#Function for hardcode prediction based on instituional studies
@app.route("/harvard", methods=['POST','GET'])
def harvard():
    
    if request.method == 'POST':  

        resignation_rate = float(request.form['resignation_rate'])
        industry = float(request.form['industry'])
        time_betweeen_promotions = float(request.form['time_betweeen_promotions'])
        hike = float(request.form['hike'])

        tenure = float(request.form['tenure'])
        performance = float(request.form['performance'])
        training_opportunities = float(request.form['training_opportunities'])


        result = []
        
        if resignation_rate >= 1 and resignation_rate <= 8:
            a = "The Percentage of Employee Attrition is Low Based on Below parameters"
            i = "The resignation rate is low per year in a company"
            print(a)
            
            #heading.append(a)
            result.append(i)
            
            if industry == 3 or industry == 4 or industry == 5:
                b = "There are less chances of attrition in Medical, Aviation, Automobile insdusty"
                result.append(b)
            
            if time_betweeen_promotions == 1:
                c= "The gap between two promotions of the employee is less"
                result.append(c)
                
            if hike >=11 and hike <=20:
                d = "The Employee is satisfied with the Hike"
                result.append(d)
                
            if tenure >=11 and tenure <=25:
                e = "The working tenure of employee is more"
                result.append(e)
                
            if performance == 3 or performance == 4 or performance == 5:    
                f = "The performance of the employee is High"
                result.append(f)
                
            if training_opportunities == 1:
                g = "The company give training opportunities"
                result.append(g)
                
        #################################################################################################
                
        if resignation_rate >= 9 and resignation_rate <= 15:
            a = "The Percentage of Employee Attrition is High Based on Below parameters"
            i = "The resignation rate is more per year in a company"
            print(a)
            
            #heading.append(a)
            result.append(i)
            
            if industry == 1 or industry == 2:
                b = "There are High chances of attrition in Tech and Healthcare insdusty"
                result.append(b)
            
            if time_betweeen_promotions == 2 or time_betweeen_promotions == 3:
                c= "The gap between two promotions of the employee is more"
                result.append(c)
                
            if hike >=5 and hike <=10:
                d = "The Employee is not satisfied with the Hike"
                result.append(d)
                
            if tenure >=1 and tenure <=10:
                e = "The working tenure of employee is less"
                result.append(e)
                
            if performance == 1 or performance == 2:    
                f = "The performance of the employee is Less"
                result.append(f)
                
            if training_opportunities == 0:
                g = "The company not give training opportunities"
                result.append(g)   

        if resignation_rate >= 1 and resignation_rate <= 8:
            return render_template('harvard.html', feature = result, heading = a )
        else:
            return render_template('harvard.html', feature = result, heading = a )

    else:
        return render_template('harvard.html')

@app.route("/cambridge", methods=['POST','GET'])
def cambridge():
    
    if request.method == 'POST':

        change_of_profession = float(request.form['change_of_profession'])
        further_education = float(request.form['further_education'])
        company_turnover = float(request.form['company_turnover'])
        transistion_of_state = float(request.form['transistion_of_state'])

        work_tasks_change = float(request.form['work_tasks_change'])
        increasing_age = float(request.form['increasing_age'])
        risk_of_wage_loss = float(request.form['risk_of_wage_loss'])

        job_satisfaction = float(request.form['job_satisfaction'])
        higher_income = float(request.form['higher_income'])

        working_hours = float(request.form['working_hours'])
        promotion = float(request.form['promotion'])
        job_security = float(request.form['job_security'])
        performance = float(request.form['performance'])

        result = []
        
        if change_of_profession == 1:
            a = "The Percentage of Employee Attrition is High Based on Below parameters"
            i = "The profession of the employee is changed"
            print(a)
            
            #heading.append(a)
            result.append(i)
            if increasing_age == 1:
                b = "The Employee age is increasing"
                result.append(b)
                
            if transistion_of_state ==1:
                c= "The Employee change the state for better job"
                result.append(c)
                
            if job_satisfaction == 0:
                d = "The Employee is not satified with the job"
                result.append(d)
                
            if higher_income == 0:
                e = "The income of the employee is not good."
                result.append(e)
                
            if working_hours == 1:    
                f = "The employee do over-time"
                result.append(f)
                
            if company_turnover == 0:
                g = "The turnover/profit of the company is in loss"
                result.append(g)
                
            if further_education == 1:
                h = "The Employee left for further studies"
                result.append(h)
                
            if work_tasks_change == 1:
                j = "The task of the employee changed suddenly"
                result.append(j)
                
            if risk_of_wage_loss == 1:
                g = "The wages of the employee is reduced"
                result.append(g)
                
            if promotion == 0:
                h = "The Employee is not promoted"
                result.append(h)
                
            if job_security == 0:
                j = "The job security is less"
                result.append(j)
                
            if performance == 1 or performance == 2:
                g = "The performance of the employee is less"
                result.append(g)
                
        #################################################################################################
                
        if change_of_profession == 0:
            a = "The Percentage of Employee Attrition is Low Based on Below parameters"
            i = "The profession of the employee is not changed"
            print(a)
                    
            #heading.append(a)
            result.append(i)
            if increasing_age == 0:
                b = "The Employee age is as per the experience"
                result.append(b)
                
            if transistion_of_state == 0:
                c= "The Employee is in the same state"
                result.append(c)
                
            if job_satisfaction == 1:
                d = "The Employee is satified with the job"
                result.append(d)
                
            if higher_income == 1:
                e = "The income of the employee is good"
                result.append(e)
                
            if working_hours == 0:    
                f = "The employee work in working hours"
                result.append(f)
                
            if company_turnover == 1:
                g = "The turnover/profit of the company is in profit"
                result.append(g)
                
            if further_education == 0:
                h = "The Employee managing further studies with job"
                result.append(h)
                
            if work_tasks_change == 0:
                j = "The task of the employee is not changed till its completion"
                result.append(j)
                
            if risk_of_wage_loss == 0:
                g = "The wages of the employee is increasing based on performance"
                result.append(g)
                
            if promotion == 1:
                h = "The Employee is promoted annually"
                result.append(h)
                
            if job_security == 1:
                j = "The job security is High"
                result.append(j)
                
            if performance == 3 or performance == 4 or performance == 5:
                g = "The performance of the employee is High"
                result.append(g)

        if change_of_profession == 0:
            return render_template('cambridge.html', feature = result, heading = a )
        else:
            return render_template('cambridge.html', feature = result, heading = a )

    else:
        return render_template('cambridge.html')

#function to do prediction based on user input file after uploading
@app.route("/ml", methods=['POST','GET'])
def ml():
    
    cleandir("static/uploads")
    if request.method == 'POST':
        file = request.files['csvfile']
        filename = secure_filename(file.filename)

        

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        

        df = pd.read_csv('static/uploads' + "/"+ filename)
        
        shape = df.shape

        cat_col = df.select_dtypes(['object']).columns

        object_list = []

        for obj in cat_col:
            object_list.append(obj)

        label_encoder = preprocessing.LabelEncoder()
        for encode in object_list:
        #print(encode)
            df[encode]= label_encoder.fit_transform(df[encode])

        X = df.drop("Attrition",axis=1)
        Y = df['Attrition']

        # Split the dataset into 75% Training set and 25% Testing set
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

        #Use Random Forest Classification algorithm
        
        forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        forest.fit(X_train, Y_train)


        return render_template('ml.html', shape = shape)    

    else:
        return render_template('ml.html')

@app.route('/userinput',methods=['POST'])
def userinput():

    
    if request.method == 'POST':
        #EmployeeID = 10
        Age = float(request.form['Age'])
        MonthlyIncome = float(request.form['MonthlyIncome'])
        TotalWorkingYears = float(request.form['TotalWorkingYears'])
        DistanceFromHome = float(request.form['DistanceFromHome'])

        PercentSalaryHike = float(request.form['PercentSalaryHike'])
        JobRole = float(request.form['JobRole'])
        NumCompaniesWorked = float(request.form['NumCompaniesWorked'])

        City = float(request.form['City'])
        Business_type = float(request.form['Business_type'])

        YearsSinceLastPromotion = float(request.form['YearsSinceLastPromotion'])
        Education = float(request.form['Education'])
        MaritalStatus = float(request.form['MaritalStatus'])
        JobLevel = float(request.form['JobLevel'])

        EducationField = float(request.form['EducationField'])
        BusinessTravel = float(request.form['BusinessTravel'])
        Department = float(request.form['Department'])
        Gender = float(request.form['Gender'])

        pred = model1.predict(np.array([[Age, MonthlyIncome, TotalWorkingYears, DistanceFromHome, 
                                        PercentSalaryHike, JobRole, NumCompaniesWorked, 
                                        YearsSinceLastPromotion, Education, MaritalStatus,  Business_type, City,
                                        JobLevel, EducationField, BusinessTravel, Department, Gender]]))

            
        #return render_template('ml.html', feature = "Hello")
        print(pred)
        if pred == 0:
            return render_template('ml.html', feature = "The Employee will not leave the company")
        else:
            return render_template('ml.html', feature = "The Employee will leave the company")
    return render_template('ml.html')


if __name__=="__main__":
    port = int(os.environ.get("PORT", 9044))
    app.run(host='0.0.0.0', port=port) 
    #app.run(debug=True)

