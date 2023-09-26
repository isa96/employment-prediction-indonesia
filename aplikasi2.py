import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import PolynomialFeatures

title = 'Application for predicting employed and unemployed individuals in Indonesia👷‍♀️👷🏢'
subtitle = 'Predict the number of employed and unemployed individuals in Indonesia using Machine Learning👷‍♀️👷🏢 '

def main():
    st.set_page_config(layout="centered", page_icon='🏢💻', page_title='Lets Predicting the number of workers and employed population')
    st.title(title)
    st.write(subtitle)
    st.write("For more information about this project, check here: [GitHub Repo](https://github.com/PrastyaSusanto/Employment-Prediction-in-Indonesia/tree/main)")

    form = st.form("Data Input")
    options = ['Employed', 'Unemployed']
    selected_options = form.multiselect('Employed/Unemployed', options, default=options)
    start_date = form.date_input('Start Date')
    end_date = form.date_input('End Date')

    submit = form.form_submit_button("Predict")  # Add a submit button

    if submit:
        # Create a list to store the selected 'Option' for each date
        option_list = []
        for date in pd.date_range(start=start_date, end=end_date):
            option_list.extend(selected_options)

        data = {
            'Tanggal Referensi': pd.date_range(start=start_date, end=end_date).repeat(len(selected_options)),
            'Option': [0 if option == 'Employed' else 1 for option in option_list]
        }
        data = pd.DataFrame(data)

        # Convert Tanggal column to datetime and calculate the difference from the reference date
        data['Tanggal Referensi'] = (data['Tanggal Referensi'] - pd.to_datetime('2011-02-01')).dt.days

        # Load the model from the pickle file
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Use PolynomialFeatures with degree 2 to transform input data
        poly = PolynomialFeatures(degree=2)
        data_poly = poly.fit_transform(data[['Tanggal Referensi', 'Option']])

        # Make prediction using the loaded model and transformed data_poly
        predictions = model.predict(data_poly)

        # Create a DataFrame to store the results
        results = pd.DataFrame({'Date': pd.date_range(start=start_date, end=end_date).repeat(len(selected_options)), 'Option': option_list, 'Predicted Number': predictions})

        # Format the Date column in the results DataFrame and remove time part
        results['Date'] = results['Date'].dt.strftime('%d-%m-%Y')

        # If only one option is selected, adjust the column name accordingly
        if len(selected_options) == 1:
            option_name = selected_options[0]
            pivot_results = results.pivot(index='Date', columns='Option', values='Predicted Number')
            pivot_results.columns = [option_name]
        else:
            # Pivot the DataFrame to create separate columns for Employed and Unemployed
            pivot_results = results.pivot(index='Date', columns='Option', values='Predicted Number')
            pivot_results.columns = ['Employed', 'Unemployed']

        # Visualize the results using two line charts or one line chart if only one option selected
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 5))
        for option in selected_options:
            plt.plot(pivot_results.index, pivot_results[option], color='royalblue')
        plt.xlabel('Date')
        plt.ylabel('Predicted Number')
        plt.xticks(rotation=90)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        if len(selected_options) == 1:
            plt.title('Predicted Amount of {} over Time'.format(option_name))
        else:
            plt.title('Predicted Amount of Employed and Unemployed over Time')
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))  # Set maximum number of x-axis ticks

        st.pyplot(plt)

        # Show the table with three columns: Date, Employed, and Unemployed
        st.dataframe(pivot_results[['Employed', 'Unemployed']])

if __name__ == '__main__':
    main()
