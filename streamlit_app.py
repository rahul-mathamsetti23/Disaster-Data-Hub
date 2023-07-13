import pandas as pd
import altair as alt
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# This code provides a selection of interactive visualizations for examining data on global disasters using Altair and Plotly charts in a Streamlit interface. By selecting a nation, a year, or both, you can explore graphs that indicate the number and different kinds of disasters. The visualizations provide a simple, entertaining, and interactive way to understand the patterns and events of significant global disasters.

def page_all_disasters():

    df = pd.read_csv("./data/Main.csv")
    countries = df['Country'].unique()

    st.write(f"## Total disasters for a specific country")

    selected_country1 = st.selectbox("Select a country for chart 1", countries, key='chart1')
    country_data = df[df['Country'] == selected_country1].drop('Total', axis=1)
    melted_data = pd.melt(country_data, id_vars=['ObjectId', 'Country', 'Indicator'], var_name='Year', value_name='Total')
    chart1 = alt.Chart(melted_data[melted_data['Indicator'] != 'TOTAL']).mark_bar().encode(
        x=alt.X('Year:N', title='Year'),
        y=alt.Y('Total:Q', title='Total'),
        color='Indicator:N',
        tooltip=['Year', 'Total', 'Indicator']
    ).properties(
        width=800,
        height=500,
        title=f"Country - {selected_country1}"
    )
    st.altair_chart(chart1)

    st.write(f"## Trend of total disasters for a specific country")

    selected_country2 = st.selectbox("Select a country for chart 1", countries, key='chart2')
    country_data = df[df['Country'] == selected_country2].drop('Total', axis=1)
    melted_data = pd.melt(country_data, id_vars=['ObjectId', 'Country', 'Indicator'], var_name='Year', value_name='Total')
    melted_data = melted_data[melted_data['Indicator'] != 'TOTAL']
    chart2 = alt.Chart(melted_data).mark_line().encode(
        x=alt.X('Year:N', title='Year'),
        y=alt.Y('Total:Q', title='Total'),
        color='Indicator:N',
        tooltip=['Year', 'Total', 'Indicator']
    ).properties(
        width=800,
        height=500,
        title=f"Country -  {selected_country2}"
    )
    st.altair_chart(chart2)

    st.write(f"## Number of Disasters in a Selected Country Over the Last Two Decades")

    df = pd.read_csv("./data/Main.csv")
    countries = df['Country'].unique()
    selected_country = st.selectbox("Select a country", countries, key='country_select')
    country_data = df[df['Country'] == selected_country]
    country_data = country_data[country_data['Indicator'] != 'TOTAL']
    bar_chart = alt.Chart(country_data).mark_bar().encode(
        x=alt.X('Indicator:N', sort='-x'),
        y=alt.Y('Total:Q', axis=alt.Axis(title='Occurrences')),
        tooltip=['Indicator', 'Total']
    ).properties(
        width=300,
        height=200,
    )
    pie_chart = alt.Chart(country_data).mark_arc().encode(
        theta='Total:Q',
        color='Indicator:N',
        tooltip=['Indicator', 'Total']
    ).properties(
        width=300,
        height=200,
    )
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write(alt.hconcat(bar_chart, pie_chart))

    df = pd.read_csv("./data/Main.csv")
    years = [str(year) for year in range(2001, 2022)]

    st.write(f"## Distribution of types of disasters across all countries for a specific year")

    selected_year = st.selectbox("Select a year", years)
    year_data = df[['Country', 'Indicator', selected_year]]
    grouped_data = year_data.groupby('Indicator').sum().reset_index()
    grouped_data = grouped_data[grouped_data['Indicator'] != 'TOTAL']
    chart = alt.Chart(grouped_data).mark_arc().encode(
        theta=selected_year,
        color='Indicator:N',
        tooltip=['Indicator', selected_year]
    ).properties(
        width=700,
        height=400,
        title=f"Distribution of types of disasters across all countries in {selected_year}"
    )
    st.altair_chart(chart)
    total = year_data[selected_year].sum()
    st.write(f"Total occurrences of all types of disasters in all countries in {selected_year}: {total}")
    
    total_data = df[df['Indicator'] == 'TOTAL'].reset_index()
    st.write(f"## Total occurrences of disasters by country")
    map_data = total_data.groupby('Country')['Total'].sum().reset_index()
    fig = px.choropleth(map_data, locations='Country', locationmode='country names',
                        color='Total', range_color=(0, map_data['Total'].max()),
                        width=800, height=600)
    st.plotly_chart(fig)
    
    df_original = pd.read_csv("./data/Original.csv")

    df_cleaned = pd.read_csv("./data/Main.csv")

    selected_dataset = st.radio("Select dataset", ("Original", "Cleaned"))
    if selected_dataset == "Original":
        st.write("# Original Dataset")
        st.write(df_original)
    else:
        st.write("# Cleaned Dataset")
        st.write(df_cleaned)

    
# This code implements an ARIMA model to forecast the probability of natural disasters in a certain country and disaster type over the next five years. The user selects the country and type of disaster from a menu before clicking a button to generate the forecast. The forecasts are displayed using an Altair line chart.


def prediction():
  
    def fit_and_forecast_arima(data, country, indicator):
        filtered_data = data[(data['Country'] == country) & (data['Indicator'] == indicator)]
        filtered_data = filtered_data[years].T
        filtered_data.index = pd.to_datetime(filtered_data.index, format='%Y')

        try:
            arima_model = ARIMA(filtered_data, order=(1, 1, 1))
            arima_results = arima_model.fit()
            forecast_arima = arima_results.forecast(steps=5)
        except ValueError:
            forecast_arima = pd.Series([0] * 5, index=pd.date_range(start=filtered_data.index[-1] + pd.DateOffset(years=1), periods=5, freq='AS'))

        return forecast_arima

    data = pd.read_csv('./data/Main.csv')
    years = [str(x) for x in range(2001, 2022)]

    st.title('Natural Disaster Prediction')
    st.write('Select a country and disaster type to forecast occurrences in the next 5 years.')

    countries = data['Country'].unique().tolist()
    disasters = data[data['Indicator'] != 'TOTAL']['Indicator'].unique().tolist()
    selected_country = st.selectbox('Country:', countries, index=countries.index('United States'))
    selected_disaster = st.selectbox('Disaster Type:', disasters, index=disasters.index('Storm'))

    if st.button('Get Prediction'):
        forecast_arima = fit_and_forecast_arima(data, selected_country, selected_disaster)

        st.subheader(f'ARIMA Predictions for {selected_country} - {selected_disaster}')
        chart_data = pd.DataFrame({
            'Year': forecast_arima.index.year,
            'Predictions': forecast_arima.values
        })

        chart = alt.Chart(chart_data).mark_line().encode(
            alt.X('Year:O', axis=alt.Axis(title='Year')),
            alt.Y('Predictions:Q', axis=alt.Axis(title='Predictions'))
        )

        st.altair_chart(chart, use_container_width=True)

        
# This code enables interactive exploration of drought data, such as the frequency and number of droughts by country and year. Users can explore various charts including a choropleth map, a bubble chart, and a pie chart by selecting countries from a dropdown menu in addition to viewing a bar chart showing frequency through time. These visualizations offer a simple means to understand patterns and trends in drought data.

def page_second():

    df = pd.read_csv("./data/Drought.csv")
    
    countries = df['Country'].unique()
    
    st.write("# Drought Frequency by Country")

    selected_countries = st.multiselect("Select countries", countries, default=["United States", "India"])

    country_data = df[df['Country'].isin(selected_countries)].drop(['Total','ObjectId'], axis=1)

    melted_data = pd.melt(country_data, id_vars=['Country', 'Indicator'], var_name='Year', value_name='Drought Frequency')

    chart = alt.Chart(melted_data[melted_data['Indicator'] == 'Drought']).mark_bar().encode(
        x=alt.X('Year:N', title='Year', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Drought Frequency:Q', title='Drought Frequency'),
        color=alt.Color('Country:N', legend=alt.Legend(title="Country")),
    ).properties(
        width=800,
        height=500,
        title="Drought Frequency by Country"
    )

    st.altair_chart(chart)
    
    ###############################################################
    
    st.write(f"### Geographical Distribution of Drought Occurrences Among Various Countries")
    fig = px.choropleth(data_frame=df,
                    locations='Country',
                    locationmode='country names',
                    color='Total',
                    scope='world')

    st.plotly_chart(fig)

    ###############################################################

    st.write(f"## Drought Count by Year for a Specific Country")

    selected_country = st.selectbox("Select a Country", countries, key='chart2')

    country_data = df[df['Country'] == selected_country].drop(['Total', 'Indicator', 'ObjectId'], axis=1)

    melted_data = pd.melt(country_data, id_vars=['Country'], var_name='Year', value_name='Drought_Count')

    chart2 = alt.Chart(melted_data).mark_bar(color='brown').encode(
        x=alt.X('Year:N', title='Year'),
        y=alt.Y('Drought_Count:Q', title='Drought Count'),
        tooltip=['Year', 'Drought_Count']
    ).properties(
        width=600,
        height=500,
        title=f"Country selected - {selected_country}"
    )

    st.altair_chart(chart2)
    
    ###############################################################

    drought_data = df[df['Indicator'] == 'Drought'].groupby(['Country']).sum().reset_index()

    chart = alt.Chart(drought_data).mark_circle().encode(
        x=alt.X('Country:N', sort='-y'),
        y=alt.Y('Total:Q', title='Total Number of Droughts'),
        color=alt.Color('Country:N', legend=None),
        size=alt.Size('Total:Q', legend=None),
        tooltip=['Country', 'Total']
    ).properties(
        width=700,
        height=500
    ).interactive()
    
    st.write(f"### Proportion of Total Number of Droughts by Country")

    st.altair_chart(chart)
    
    ###############################################################

    data = pd.read_csv('./data/Drought.csv')

    total_occurrences = data["Total"].sum()

    year_columns = [str(year) for year in range(2001, 2022)]
    percentages = [data[year].sum() / total_occurrences * 100 for year in year_columns]

    df = pd.DataFrame({"Year": year_columns, "Percentage": percentages})

    st.write(f"### Contribution of Each Year's Drought Occurrences to the Total Number of Droughts")

    fig = px.pie(df, values="Percentage", names="Year")

    st.plotly_chart(fig)
    
    
# This code provides several interactive visualizations for examining data on extreme temperatures, such as the frequency and count of extreme temperatures by country and year. Users can select a country from a dropdown menu to view a bar chart that shows frequency over time. They can also explore other charts, such as a pie chart, a choropleth map, and bubble charts. These illustrations make it simple to comprehend the patterns and trends in the data on extreme temperatures.

    
def page_third():
    
    df = pd.read_csv("./data/Extreme_temperature.csv")

    countries = df['Country'].unique()

    st.write("# Extreme Temperature Frequency by Country")

    selected_countries = st.multiselect("Select countries", countries, default=["United States", "India"])

    country_data = df[df['Country'].isin(selected_countries)].drop(['Total','ObjectId'], axis=1)

    melted_data = pd.melt(country_data, id_vars=['Country', 'Indicator'], var_name='Year', value_name='Frequency')

    chart = alt.Chart(melted_data[melted_data['Indicator'] == 'Extreme temperature']).mark_bar().encode(
        x=alt.X('Year:N', title='Year', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Frequency:Q', title='Frequency'),
        color=alt.Color('Country:N', legend=alt.Legend(title="Country")),
    ).properties(
        width=800,
        height=500,
        title="Extreme Temperature Frequency by Country"
    )

    st.altair_chart(chart)

    ###############################################################
    
    st.write(f"### Geographical Distribution of Extreme Temperature Occurrences Among Various Countries")
    fig = px.choropleth(data_frame=df,
                    locations='Country',
                    locationmode='country names',
                    color='Total',
                    scope='world')

    st.plotly_chart(fig)

    ###############################################################

    st.write(f"## Extreme Temperature Count by Year for a Specific Country")

    selected_country = st.selectbox("Select a Country", countries, key='chart2')

    country_data = df[df['Country'] == selected_country].drop(['Total', 'Indicator', 'ObjectId'], axis=1)

    melted_data = pd.melt(country_data, id_vars=['Country'], var_name='Year', value_name='Count')

    chart2 = alt.Chart(melted_data).mark_bar(color='red').encode(
        x=alt.X('Year:N', title='Year'),
        y=alt.Y('Count:Q', title='Count'),
        tooltip=['Year', 'Count']
    ).properties(
        width=600,
        height=500,
        title=f"Country selected - {selected_country}"
    )

    st.altair_chart(chart2)

    ###############################################################

    temperature_data = df[df['Indicator'] == 'Extreme temperature'].groupby(['Country']).sum().reset_index()

    chart = alt.Chart(temperature_data).mark_circle().encode(
        x=alt.X('Country:N', sort='-y'),
        y=alt.Y('Total:Q', title='Total Number of Extreme Temperatures'),
        color=alt.Color('Country:N', legend=None),
        size=alt.Size('Total:Q', legend=None),
        tooltip=['Country', 'Total']
    ).properties(
        width=700,
        height=500
    ).interactive()

    st.write(f"### Proportion of Total Number of Extreme Temperatures by Country")

    st.altair_chart(chart)

    ###############################################################

    data = pd.read_csv('./data/Extreme_temperature.csv')

    total_occurrences = data["Total"].sum()

    year_columns = [str(year) for year in range(2001, 2022)]
    percentages = [data[year].sum() / total_occurrences * 100 for year in year_columns]

    df = pd.DataFrame({"Year": year_columns, "Percentage": percentages})

    st.write(f"### Contribution of Each Year's Extreme Temperature Occurrences to the Total Number of Extreme Temperatures")

    fig = px.pie(df, values="Percentage", names="Year")

    st.plotly_chart(fig)
    

# The "page_fourth" function in this application loads and shows flood statistics broken down by country. Interactive visualizations, such as bar charts and choropleth maps, created using Plotly and Altair, such as the frequency and number of floods in various countries. A pie chart that shows the percentage contribution of each year to the total frequency of the flood indicator occurrences is another element of the function.


def page_fourth():
    
    df = pd.read_csv("./data/Flood.csv")

    countries = df['Country'].unique()

    st.write("# Flood Frequency by Country")

    selected_countries = st.multiselect("Select countries", countries, default=["United States", "India"])

    country_data = df[df['Country'].isin(selected_countries)].drop(['Total','ObjectId'], axis=1)

    melted_data = pd.melt(country_data, id_vars=['Country', 'Indicator'], var_name='Year', value_name='Flood Frequency')

    chart = alt.Chart(melted_data[melted_data['Indicator'] == 'Flood']).mark_bar().encode(
        x=alt.X('Year:N', title='Year', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Flood Frequency:Q', title='Flood Frequency'),
        color=alt.Color('Country:N', legend=alt.Legend(title="Country")),
    ).properties(
        width=800,
        height=500,
        title="Flood Frequency by Country"
    )

    st.altair_chart(chart)

    ###############################################################

    st.write(f"### Geographical Distribution of Flood Occurrences Among Various Countries")
    fig = px.choropleth(data_frame=df,
                        locations='Country',
                        locationmode='country names',
                        color='Total',
                        scope='world')

    st.plotly_chart(fig)

    ###############################################################

    st.write(f"## Flood Count by Year for a Specific Country")

    selected_country = st.selectbox("Select a Country", countries, key='chart2')

    country_data = df[df['Country'] == selected_country].drop(['Total', 'Indicator', 'ObjectId'], axis=1)

    melted_data = pd.melt(country_data, id_vars=['Country'], var_name='Year', value_name='Flood_Count')

    chart2 = alt.Chart(melted_data).mark_bar(color='blue').encode(
        x=alt.X('Year:N', title='Year'),
        y=alt.Y('Flood_Count:Q', title='Flood Count'),
        tooltip=['Year', 'Flood_Count']
    ).properties(
        width=600,
        height=500,
        title=f"Country selected - {selected_country}"
    )

    st.altair_chart(chart2)

    ###############################################################

    flood_data = df[df['Indicator'] == 'Flood'].groupby(['Country']).sum().reset_index()

    chart = alt.Chart(flood_data).mark_circle().encode(
        x=alt.X('Country:N', sort='-y'),
        y=alt.Y('Total:Q', title='Total Number of Floods'),
        color=alt.Color('Country:N', legend=None),
        size=alt.Size('Total:Q', legend=None),
        tooltip=['Country', 'Total']
    ).properties(
        width=700,
        height=500
    ).interactive()

    st.write(f"### Proportion of Total Number of Floods by Country")

    st.altair_chart(chart)

    ###############################################################

    data = pd.read_csv('./data/Flood.csv')

    total_occurrences = data["Total"].sum()

    year_columns = [str(year) for year in range(2001, 2022)]
    percentages = [data[year].sum() / total_occurrences * 100 for year in year_columns]

    df = pd.DataFrame({"Year": year_columns, "Percentage": percentages})

    st.write(f"### Contribution of Each Year's Flood Occurrences to the Total Number of Floods")

    fig = px.pie(df, values="Percentage", names="Year")

    st.plotly_chart(fig)
    

# The fifth page of this website application examines natural disasters. A choropleth map is used to display how frequently landslides occur in different countries and years, and the computer analyzes data on landslides to produce these maps. It also shows the total number of landslides by nation and the proportion that each year adds to the overall total using pie charts.

    
def page_fifth():
    
    df = pd.read_csv("./data/Landslide.csv")

    countries = df['Country'].unique()

    st.write("# Landslide Frequency by Country")

    selected_countries = st.multiselect("Select countries", countries, default=["United States", "India"])

    country_data = df[df['Country'].isin(selected_countries)].drop(['Total','ObjectId'], axis=1)

    melted_data = pd.melt(country_data, id_vars=['Country', 'Indicator'], var_name='Year', value_name='Landslide Frequency')

    chart = alt.Chart(melted_data[melted_data['Indicator'] == 'Landslide']).mark_bar().encode(
        x=alt.X('Year:N', title='Year', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Landslide Frequency:Q', title='Landslide Frequency'),
        color=alt.Color('Country:N', legend=alt.Legend(title="Country")),
    ).properties(
        width=800,
        height=500,
        title="Landslide Frequency by Country"
    )

    st.altair_chart(chart)

    ###############################################################

    st.write(f"### Geographical Distribution of Landslide Occurrences Among Various Countries")
    fig = px.choropleth(data_frame=df,
                    locations='Country',
                    locationmode='country names',
                    color='Total',
                    scope='world')

    st.plotly_chart(fig)

    ###############################################################

    st.write(f"## Landslide Count by Year for a Specific Country")

    selected_country = st.selectbox("Select a Country", countries, key='chart2')

    country_data = df[df['Country'] == selected_country].drop(['Total', 'Indicator', 'ObjectId'], axis=1)

    melted_data = pd.melt(country_data, id_vars=['Country'], var_name='Year', value_name='Landslide_Count')

    chart2 = alt.Chart(melted_data).mark_bar(color='yellow').encode(
        x=alt.X('Year:N', title='Year'),
        y=alt.Y('Landslide_Count:Q', title='Landslide Count'),
        tooltip=['Year', 'Landslide_Count']
    ).properties(
        width=600,
        height=500,
        title=f"Country selected - {selected_country}"
    )

    st.altair_chart(chart2)

    ###############################################################

    landslide_data = df[df['Indicator'] == 'Landslide'].groupby(['Country']).sum().reset_index()

    chart = alt.Chart(landslide_data).mark_circle().encode(
        x=alt.X('Country:N', sort='-y'),
        y=alt.Y('Total:Q', title='Total Number of Landslides'),
        color=alt.Color('Country:N', legend=None),
        size=alt.Size('Total:Q', legend=None),
        tooltip=['Country', 'Total']
    ).properties(
        width=700,
        height=500
    ).interactive()

    st.write(f"### Proportion of Total Number of Landslides by Country")

    st.altair_chart(chart)

    ###############################################################

    data = pd.read_csv('./data/Landslide.csv')

    total_occurrences = data["Total"].sum()

    year_columns = [str(year) for year in range(2001, 2022)]
    percentages = [data[year].sum() / total_occurrences * 100 for year in year_columns]

    df = pd.DataFrame({"Year": year_columns, "Percentage": percentages})

    st.write(f"### Contribution of Each Year's Landslide Occurrences to the Total Number of Landslides")

    fig = px.pie(df, values="Percentage", names="Year")

    st.plotly_chart(fig)

    
# The page_sixth function pulls data on storm frequency by country from a CSV file and displays it in several charts. Users may browse statistics on storm frequency and count by year while choosing one or more nations. Along with a choropleth map showing the locations of the storms, the function also includes a chart showing the percentage of total storms per country.

    
def page_sixth():


    df = pd.read_csv("./data/Storm.csv")

    countries = df['Country'].unique()

    st.write("# Storm Frequency by Country")

    selected_countries = st.multiselect("Select countries", countries, default=["United States", "India"])

    country_data = df[df['Country'].isin(selected_countries)].drop(['Total','ObjectId'], axis=1)

    melted_data = pd.melt(country_data, id_vars=['Country', 'Indicator'], var_name='Year', value_name='Storm Frequency')

    chart = alt.Chart(melted_data[melted_data['Indicator'] == 'Storm']).mark_bar().encode(
        x=alt.X('Year:N', title='Year', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Storm Frequency:Q', title='Storm Frequency'),
        color=alt.Color('Country:N', legend=alt.Legend(title="Country")),
    ).properties(
        width=800,
        height=500,
        title="Storm Frequency by Country"
    )

    st.altair_chart(chart)

    ###############################################################

    st.write(f"### Geographical Distribution of Storm Occurrences Among Various Countries")
    fig = px.choropleth(data_frame=df,
                    locations='Country',
                    locationmode='country names',
                    color='Total',
                    scope='world')

    st.plotly_chart(fig)

    ###############################################################

    st.write(f"## Storm Count by Year for a Specific Country")

    selected_country = st.selectbox("Select a Country", countries, key='chart2')

    country_data = df[df['Country'] == selected_country].drop(['Total', 'Indicator', 'ObjectId'], axis=1)

    melted_data = pd.melt(country_data, id_vars=['Country'], var_name='Year', value_name='Storm_Count')

    chart2 = alt.Chart(melted_data).mark_bar(color='purple').encode(
        x=alt.X('Year:N', title='Year'),
        y=alt.Y('Storm_Count:Q', title='Storm Count'),
        tooltip=['Year', 'Storm_Count']
    ).properties(
        width=600,
        height=500,
        title=f"Country selected - {selected_country}"
    )

    st.altair_chart(chart2)

    ###############################################################

    storm_data = df[df['Indicator'] == 'Storm'].groupby(['Country']).sum().reset_index()

    chart = alt.Chart(storm_data).mark_circle().encode(
        x=alt.X('Country:N', sort='-y'),
        y=alt.Y('Total:Q', title='Total Number of Storms'),
        color=alt.Color('Country:N', legend=None),
        size=alt.Size('Total:Q', legend=None),
        tooltip=['Country', 'Total']
    ).properties(
        width=700,
        height=500
    ).interactive()

    st.write(f"### Proportion of Total Number of Storms by Country")

    st.altair_chart(chart)

    ###############################################################

    data = pd.read_csv('./data/Storm.csv')

    total_occurrences = data["Total"].sum()

    year_columns = [str(year) for year in range(2001, 2022)]
    percentages = [data[year].sum() / total_occurrences * 100 for year in year_columns]

    df = pd.DataFrame({"Year": year_columns, "Percentage": percentages})

    st.write(f"### Contribution of Each Year's Storm Occurrences to the Total Number of Storms")

    fig = px.pie(df, values="Percentage", names="Year")

    st.plotly_chart(fig)
    

# The application loads a dataset on wildfire occurrences and shows graphs illustrating their frequency and geographic distribution. The code also displays the proportion of all wildfires by country and the percentage contribution of each year to the overall number of occurrences. Users can select a country and view the wildfire counts by year. The code is repeated for each type of natural disaster (flood, landslide, and storm), each of which includes a different set of visuals.

    
def page_seventh():
    

    df = pd.read_csv("./data/Wildfire.csv")

    countries = df['Country'].unique()

    st.write("# Wildfire Frequency by Country")

    selected_countries = st.multiselect("Select countries", countries, default=["United States", "India"])

    country_data = df[df['Country'].isin(selected_countries)].drop(['Total','ObjectId'], axis=1)

    melted_data = pd.melt(country_data, id_vars=['Country', 'Indicator'], var_name='Year', value_name='Wildfire Frequency')

    chart = alt.Chart(melted_data[melted_data['Indicator'] == 'Wildfire']).mark_bar().encode(
        x=alt.X('Year:N', title='Year', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Wildfire Frequency:Q', title='Wildfire Frequency'),
        color=alt.Color('Country:N', legend=alt.Legend(title="Country")),
    ).properties(
        width=800,
        height=500,
        title="Wildfire Frequency by Country"
    )

    st.altair_chart(chart)

    ###############################################################

    st.write(f"### Geographical Distribution of Wildfire Occurrences Among Various Countries")
    fig = px.choropleth(data_frame=df,
                    locations='Country',
                    locationmode='country names',
                    color='Total',
                    scope='world')

    st.plotly_chart(fig)

    ###############################################################

    st.write(f"## Wildfire Count by Year for a Specific Country")

    selected_country = st.selectbox("Select a Country", countries, key='chart2')

    country_data = df[df['Country'] == selected_country].drop(['Total', 'Indicator', 'ObjectId'], axis=1)

    melted_data = pd.melt(country_data, id_vars=['Country'], var_name='Year', value_name='Wildfire_Count')

    chart2 = alt.Chart(melted_data).mark_bar(color='orange').encode(
        x=alt.X('Year:N', title='Year'),
        y=alt.Y('Wildfire_Count:Q', title='Wildfire Count'),
        tooltip=['Year', 'Wildfire_Count']
    ).properties(
        width=600,
        height=500,
        title=f"Country selected - {selected_country}"
    )

    st.altair_chart(chart2)

    ###############################################################

    wildfire_data = df[df['Indicator'] == 'Wildfire'].groupby(['Country']).sum().reset_index()

    chart = alt.Chart(wildfire_data).mark_circle().encode(
        x=alt.X('Country:N', sort='-y'),
        y=alt.Y('Total:Q', title='Total Number of Wildfires'),
        color=alt.Color('Country:N', legend=None),
        size=alt.Size('Total:Q', legend=None),
        tooltip=['Country', 'Total']
    ).properties(
        width=700,
        height=500
    ).interactive()

    st.write(f"### Proportion of Total Number of Wildfires by Country")

    st.altair_chart(chart)

    ###############################################################

    data = pd.read_csv('./data/Wildfire.csv')

    total_occurrences = data["Total"].sum()

    year_columns = [str(year) for year in range(2001, 2022)]
    percentages = [data[year].sum() / total_occurrences * 100 for year in year_columns]

    df = pd.DataFrame({"Year": year_columns, "Percentage": percentages})

    st.write(f"### Contribution of Each Year's Wildfire Occurrences to the Total Number of Wildfires")

    fig = px.pie(df, values="Percentage", names="Year")

    st.plotly_chart(fig)

    
def main():
    
    st.set_page_config(page_title="Disaster Data Hub")
    st.sidebar.title("Navigation")
    
    pages = {
        
        "Disaster Analytics": page_all_disasters,
        "Future Prediction" : prediction,
        "Drought Analysis": page_second,
        "Extreme Temperature Analysis": page_third,
        "Flood Analysis": page_fourth,
        "Landslide Analysis": page_fifth,
        "Storm Analysis": page_sixth,
        "Wildfire Analysis": page_seventh,
 
    }
    
    page = st.sidebar.selectbox("Main Menu", tuple(pages.keys()))
    pages[page]()

if __name__ == "__main__":
    main()
