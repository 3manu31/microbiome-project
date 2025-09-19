# 🦠 Microbiome Explorer Dashboard

This project started as a personal learning journey to understand microbiome analysis through coding and create a tool that’s useful for researchers.

It is an interactive web application for exploring and visualizing microbiome data patterns across different demographic and health groups. Built for researchers studying gut microbiome diversity and abundance patterns.

## 🚀 Live Demo

[**Try the live demo**](https://microbiome-project.streamlit.app/) - Uses distilled sample data from the GMRepo dataset (PRJEB11419)

## ✨ Key Features

- **Interactive Visualizations**: Compare microbe abundance across multiple groups
- **Smart Microbe Labeling**: Clean M1, M2, M3 format for easy identification  
- **Multiple Grouping Options**: Health status, mental illness, sex, sample type, ASD
- **Customizable Analysis**: Choose top N microbes (5-15) for focused analysis
- **Data Upload Support**: Use your own metadata and BIOM files (local only)
- **High-Performance Caching**: Lightning-fast chart loading for repeated analyses
- **Export-Ready Charts**: High-quality visualizations with proper legends

## 📊 What You'll See

- **Grouped Comparison Charts**: Side-by-side microbe abundance across selected groups
- **Individual Group Charts**: Detailed view of top microbes per group
- **Interactive Tables**: Numerical data with microbe ID mapping
- **Clean Interface**: Professional, research-focused design

## 🔬 Perfect For:

- Microbiome researchers and students
- Gut health studies and analysis
- Comparative microbiome research
- Educational demonstrations
- Preliminary data exploration

## ⚠️ Usage Notes

- The live demo may be slow with rapid option changes due to cloud resource limits
- Please toggle one option at a time and wait for charts to load
- File upload is automatically disabled on the cloud demo; run locally for custom data
- If you encounter errors, simply reload the page and try again
- **Note**: This is an educational/demonstration project - for production research, please validate results with established bioinformatics pipelines

### How file upload is disabled on Streamlit Cloud

File uploaders are automatically hidden when the app detects it is running on Streamlit Cloud (using multiple environment variable checks for reliable detection). The app will show a clear message when file upload is disabled and display instructions for local usage. No manual changes are needed.

## 📚 Data Source & Citation

A subset of microbiome data was obtained from the American Gut Project (Qiita Study ID: 10317, <https://qiita.ucsd.edu/study/description/10317>) and processed using custom Python scripts. All other analyses and data are original to this work. For details on the American Gut Project, see McDonald et al., 2018 (doi:10.1128/mSystems.00031-18).

## Getting Started

1. Install requirements: `pip install -r requirements.txt`
2. Run the dashboard: `streamlit run streamlit_app.py`
3. Upload your own files or use the provided example files

### About the Author
Developed by Emmanuel Gialitakis, a medical student passionate about the intersection of medicine, technology, and coding. This project began as a personal learning exploration and a showcase of how computational tools can support research across clinical and lab settings.  
I welcome feedback, collaboration, and opportunities to apply or expand these techniques in diverse research environments.

## Contact
For questions or collaboration, please reach out via email (em.gialitakis@gmail.com) or LinkedIn

## 🙏 Courtesy Request
If you find this project useful and plan to use it in your own work, I kindly ask that you notify me. This is not a license requirement, just a personal request so I can learn about its impact and possibly connect with fellow researchers. Thank you!