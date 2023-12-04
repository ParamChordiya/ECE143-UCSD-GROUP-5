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


- ***Enhancing Accountability and Transparency:*** 
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
- Temporal Analysis
- Geospatial Analysis
- Age Vs. Reason for Stopping
- Race Vs. Action Taken

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
8. 

## <div align="center"><span style="color: #3498db;">File and Folder Structure</span></div>

```text
Final Project Files/
├── main.py
│
├── plots/
│   ├── __init__.py
│   ├── models.py
│   ├── views.py
│   ├── api.py
│   ├── requirements.txt
└── README.md  
```

## <div align="center"><span style="color: #3498db;">Data Sources</span></div>

Dataset: [Police Vehicle Stops Search Details - San Diego.](https://data.sandiego.gov/datasets/police-vehicle-stops-search-details/)

## <div align="center"><span style="color: #9b59b6;">Results</span></div>

Check out the detailed analysis in our [<span style="color: #3498db;">Presentation Report</span>](reports/analysis_report.pdf).
