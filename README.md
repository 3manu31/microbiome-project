# Microbiome Top Microbes Dashboard

This interactive dashboard allows researchers to explore microbiome data, compare top microbes across health, mental illness, sex, sample type, and Autism Spectrum Disorder (ASD) groups, and visualize differences with clear, customizable charts.


## Features
- Select grouping columns (with user-friendly names)
- Choose number of top microbes to display
- View single group bar charts and grouped comparison chart
- See a comparison table for all selected microbes
- Track microbes with unique IDs
- Upload your own metadata and biom files (only available when running locally)
- Robust error handling for missing files or columns

## Limitations & Warnings

- The live demo may be slow or crash if toggling options too quickly due to Streamlit Cloud resource limits.
- Please toggle one option at a time and wait for the page to load before toggling again.
- File upload is automatically disabled on the cloud demo (Streamlit Cloud); to use this feature, install and run the app locally.
- If you see errors or the app crashes, reload the page and try again.

### How file upload is disabled on Streamlit Cloud
File uploaders are automatically hidden when the app detects it is running on Streamlit Cloud (using the `STREAMLIT_SERVER_HEADLESS` environment variable). No manual changes are needed.

## Data Source & Citation

A subset of microbiome data was obtained from the American Gut Project (Qiita Study ID: 10317, https://qiita.ucsd.edu/study/description/10317) and processed using custom Python scripts. All other analyses and data are original to this work. For details on the American Gut Project, see McDonald et al., 2018 (doi:10.1128/mSystems.00031-18).

## Getting Started
1. Install requirements: `pip install -r requirements.txt`
2. Run the dashboard: `streamlit run streamlit_dashboard.py`
3. Upload your own files or use the provided example files

## About the Author
This dashboard was developed by Emmanuel Gialitakis, a medical student with interests in computational biology, data science, research, and exploring the intersection of medicine & technology. He is passionate about making science accessible to all and welcomes users from all backgrounds.
## Contact
For questions or collaboration, please reach out via GitHub or LinkedIn
