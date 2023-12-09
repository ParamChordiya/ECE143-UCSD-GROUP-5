<div align="center">

# <span style="color: #3498db;">ECE 143 Project: Police Vehicle Stops Data Analysis</span>

<img src="UCSD-Symbol.png" alt="Data Analysis" style="width:40%;">

</div>


## <div align="center"><span style="color: #27ae60;">Team Members</span></div>
- **<span style="color: #e74c3c;">Param Chordiya</span>**
- **<span style="color: #e74c3c;">Ninad Ekbote</span>**
- **<span style="color: #e74c3c;">Divya Sri Dodla</span>**
- **<span style="color: #e74c3c;">Yi-yang Chen</span>**
- **<span style="color: #e74c3c;">Yanchen Jing</span>**

## <div align="center"><span style="color: #27ae60;">Overview</span></div>

Understanding and analyzing the temporal, geospatial and demographic distributions of police vehicle stops is a critical issue that has wide-ranging implications for community safety, law enforcement practices, and the protection of civil rights. It is essential to recognize that data analysis in this context is not just a technical endeavor but a way to shed light on potentially systemic issues that could affect people's lives. This project can serve as a powerful tool to promote transparency, accountability, and fair policing practices. By revealing patterns and trends in police vehicle stops, we can help bridge the gap between law enforcement and the communities they serve.


- <u> ***Enhancing Accountability and Transparency:*** </u>
Understanding when police vehicle stops occur, as indicated by the "Timestamp," "Stop_date," and "Stop_time," can lead to increased accountability within law enforcement agencies. By analyzing this temporal
data, we aim to shed light on potential patterns, uncover any irregularities in the timing of stops, and promote transparency in law enforcement practices.
- ***Promoting Equitable Resource Allocation:***
Identifying locations where police vehicle stops frequently occur, as related to "Service_area," is crucial for effective resource allocation. By analyzing these data points, we can help law enforcement agencies target their efforts more efficiently, ensuring that resources are deployed to areas with the greatest need, thus enhancing community safety.
- ***Advocating for Fair Policing:***
Analyzing the relationship between "Subject_age" and the "Stop_cause" can uncover insights into the fairness and appropriateness of stops. We aim to examine age-related trends to determine if certain age groups are disproportionately affected by specific reasons for stops, promoting fair and unbiased policing practices.
- ***Addressing Potential Bias and Discrimination:***
Understanding the relationship between "Subject_race" and the actions taken during stops, such as "Arrested," "Searched," "Obtained_consent," and "Property_seized," is critical for addressing potential bias and discrimination. Analyzing these data points can help identify disparities and drive actions to ensure equal treatment for all individuals, regardless of their racial background.

## <div align="center"><span style="color: #27ae60;">Analysis Overview</span></div>
Some important analysis performed can be categorized as:
- Exploratory Data Analysis
- Temporal Analysis
- Geospatial Analysis
- Demographic Analysis

## <div align="center"><span style="color: #e67e22;">Installation</span></div>

To run this project locally, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/ParamChordiya/ECE143-UCSD-GROUP-5.git
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## <div align="center"><span style="color: #3498db;">External Modules & Libraries</span></div>
1. numpy
2. pandas
3. matplotlib
4. seaborn
5. datetime
6. geopandas
7. collections

## <div align="center"><span style="color: #3498db;">File and Folder Structure</span></div>

1. dataset folder: Contains all the csv files of the dataset, which we used for data analysis.
2. notebook folder: Conatins all the notebooks of the Data cleaning, Explorary Data Analysis, Temporal Analysis, Geospatial Analysis and Demographic Analysis.
3. python files: This folder contains the .py files of Data cleaning, Explorary Data Analysis, Temporal Analysis, Geospatial Analysis and Demographic Analysis.
4. final-plot.ipynb: This notebook contains the code for all the plots given in our presenation.
5. main.py: This the .py file for main.ipynb file, which contains the consilated codes for all the analysis we have done
6. final-presentation.pdf: This is our final presenation.
7. ECE 143 - Project proposal.pdf: This is our project proposal
8. README file

```text
Files/
├── dataset/
│   ├── pd_beats_datasd/
│   │   ├── pd_beats_datasd.cpg
│   │   ├── pd_beats_datasd.dbf
│   │   ├── pd_beats_datasd.prj
│   │   ├── pd_beats_datasd.shp
│   │   ├── pd_beats_datasd.shx
│   ├── pd_divisions_datasd/
│   │   ├── pd_divisions_datasd.cpg
│   │   ├── pd_divisions_datasd.dbf
│   │   ├── pd_divisions_datasd.prj
│   │   ├── pd_divisions_datasd.shp
│   │   ├── pd_divisions_datasd.shx
│   ├── final_combined_dataset_v2.csv
│   ├── final_combined_dataset_v3.csv
│   ├── vehicle_stops_2014_datasd.csv
│   ├── vehicle_stops_2015_datasd.csv
│   ├── vehicle_stops_2016_datasd.csv
│   ├── vehicle_stops_2017_datasd.csv
│   ├── vehicle_stops_dictionary.csv
│   ├── vehicle_stops_final_datasd.csv
│   ├── vehicle_stops_race_codes.csv
│   ├── vehicle_stops_search_details_2014_datasd.csv
│   ├── vehicle_stops_search_details_2015_datasd.csv
│   ├── vehicle_stops_search_details_2016_datasd.csv
│   ├── vehicle_stops_search_details_2017_datasd.csv
├── notebooks/
│   ├── age_vs_reason_action.ipynb
│   ├── complete_data_cleaning_and_label_encoding.ipynb
│   ├── exploratory_data_analysis.ipynb
│   ├── geospatial_analysis.ipynb
│   ├── temporal_analysis.ipynb
├── python/
│   ├── age_vs_reason_action.py
│   ├── complete_data_cleaning_and_label_encoding.py
│   ├── exploratory_data_analysis.py
│   ├── geospatial_analysis.py
│   ├── temporal_analysis.py
├── ECE 143 - Project proposal.pdf
├── final-plot.ipynb
├── final-presentation.pdf
├── main.py
└── README.md  
```

## <div align="center"><span style="color: #3498db;">Data Sources</span></div>

Dataset: [Police Vehicle Stops Search Details - San Diego.](https://data.sandiego.gov/datasets/police-vehicle-stops-search-details/)

## <div align="center"><span style="color: #9b59b6;">Presentation & Results</span></div>

Check out the detailed analysis in our [<span style="color: #3498db;">Presentation</span>](final-presentation.pdf).
