from flask import Flask, request, render_template
import pandas as pd
import pickle  
from flask_mail import Mail, Message
from sklearn.preprocessing import OrdinalEncoder

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Get files
        basic_file = request.files['basic_dataset']
        attendance_file = request.files['attendance_dataset']
        tests_file = request.files['tests_dataset']
        fees_file = request.files['fees_dataset']

        # Read datasets
        basic_df = pd.read_excel(basic_file) if basic_file.filename.endswith('.xlsx') else pd.read_csv(basic_file)
        attendance_df = pd.read_excel(attendance_file) if attendance_file.filename.endswith('.xlsx') else pd.read_csv(attendance_file)
        tests_df = pd.read_excel(tests_file) if tests_file.filename.endswith('.xlsx') else pd.read_csv(tests_file)
        fees_df = pd.read_excel(fees_file) if fees_file.filename.endswith('.xlsx') else pd.read_csv(fees_file)

        # Standardize column names
        for df in [basic_df, attendance_df, tests_df, fees_df]:
            df.columns = df.columns.str.strip().str.replace(" ", "_")

        # Merge datasets
        merged_df = basic_df.merge(attendance_df, on='Student_ID') \
                            .merge(tests_df, on='Student_ID') \
                            .merge(fees_df, on='Student_ID')
        
        
        
        merged_df["Attendance_Percentage"] = (merged_df["Classes_Attended"] / merged_df["Total_Classes"]) * 100
        merged_df["Average_Test_Score"] = merged_df[["Internal_Test1_Score","Internal_Test2_Score","Internal_Test3_Score"]].mean(axis=1)
        X = merged_df[["Fee_Status", "Attendance_Percentage", "Average_Test_Score"]]
   
         # Encode Fee_Status (Paid/UnPaid → 0/1)
      
        oe = OrdinalEncoder(categories=[["UnPaid", "Paid"]])
        X["Fee_Status"] = oe.fit_transform(X[["Fee_Status"]])
        
        
    # 6. Predict with model
        with open("./model/drop_rate_model.pkl", "rb") as f:
         model = pickle.load(f)
        merged_df["Predicted_Drop_Rate"] = model.predict(X)

        merged_df.to_csv("merged_students.csv", index=False)
        
        # Store merged data in session 
        # Redirect to /students page
        return render_template('signup.html')

    return render_template('signup.html')


@app.route('/students', methods=['GET', 'POST'])
def students_page():
    if request.method== "GET":
        df = pd.read_csv("merged_students.csv")
        students = df.to_dict(orient="records")
    # Get students from session
        return render_template('students.html', students= students)
    
@app.route('/student/<student_id>')
def student_profile(student_id):
    df = pd.read_csv("merged_students.csv")
    student = df[df["Student_ID"] == student_id].to_dict(orient="records")[0]
    return render_template("profile.html", student=student)
# Email configuration (for Gmail SMTP)



if __name__ == "__main__":
    app.run(debug=True)
