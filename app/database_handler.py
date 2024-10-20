from fastapi import FastAPI

from pymongo import MongoClient

app = FastAPI()


client = MongoClient("mongodb+srv://LLM_User:User_llm01@clusterllm.saneb.mongodb.net/?retryWrites=true&w=majority&appName=ClusterLLM")
db = client['chat_db']
collection = db['chat_history']
vector_store = db['vector_store'] 



def test_vector_context():
    context = {"context":'''
         **Job Title: AI Engineer**

        **Location:** Florida (Remote or On-site options available)  
        **Job Type:** Full-Time, Permanent  
        **Department:** Artificial Intelligence / Machine Learning

         **About the Company:**  
        We are a fast-growing technology company based in Florida, specializing in developing AI-driven solutions across multiple industries, including healthcare, finance, retail, and logistics. Our mission is to harness cutting-edge artificial intelligence technologies to solve real-world challenges and drive innovation.

         **Position Overview:**  
        We are seeking a highly skilled **AI Engineer** to join our dynamic team. As an AI Engineer, you will be responsible for designing, developing, and deploying machine learning models and AI systems. You will work closely with data scientists, software engineers, and other stakeholders to create AI solutions that enhance our product offerings and optimize internal operations. The ideal candidate is someone with deep expertise in artificial intelligence, machine learning algorithms, and software development.

         **Key Responsibilities:**
        - Design, build, and deploy scalable AI/ML models to solve business problems.
        - Work with large datasets to clean, preprocess, and analyze data for model training and evaluation.
        - Collaborate with cross-functional teams to integrate AI models into existing systems and applications.
        - Research and implement state-of-the-art machine learning techniques such as supervised, unsupervised, and reinforcement learning.
        - Develop and optimize machine learning pipelines and infrastructure for high-performance computing environments.
        - Continuously monitor AI systems in production and make improvements based on performance metrics and new data.
        - Write clean, maintainable, and efficient code for deploying AI solutions in a cloud environment (e.g., AWS, GCP, or Azure).
        - Stay up-to-date with the latest developments in AI and machine learning, and advocate for best practices within the organization.

         **Required Skills and Qualifications:**

        **Education & Experience:**
        - Bachelor’s degree (Master’s or Ph.D. preferred) in Computer Science, Data Science, AI, Machine Learning, or a related field.
        - **3-5 years** of professional experience in AI/ML engineering or related roles.
        
        **Core Technical Skills:**
        - **Programming Languages:** Proficiency in Python (required), with experience in frameworks such as TensorFlow, PyTorch, Keras, or Scikit-Learn.
        - **Deep Learning:** Strong understanding of neural networks, CNNs, RNNs, GANs, and other deep learning architectures.
        - **Machine Learning Algorithms:** Experience with supervised and unsupervised learning methods (e.g., regression, classification, clustering, and dimensionality reduction).
        - **NLP:** Hands-on experience with natural language processing (NLP) techniques and frameworks (e.g., spaCy, NLTK, Hugging Face Transformers).
        - **Computer Vision:** Experience with image processing and computer vision technologies, including OpenCV and deep learning-based vision models.
        - **Cloud Platforms:** Experience with cloud-based machine learning services (AWS Sagemaker, Google AI, Azure ML).
        - **Data Science Tools:** Expertise in using data manipulation libraries such as Pandas, NumPy, and visualization tools like Matplotlib or Seaborn.
        - **Databases:** Experience working with SQL and NoSQL databases (e.g., PostgreSQL, MongoDB).
        - **Version Control:** Strong knowledge of Git for version control and collaboration.
        
        **Model Deployment & Optimization:**
        - Experience with deploying AI/ML models in production environments (Docker, Kubernetes, or similar).
        - Knowledge of model optimization techniques (e.g., quantization, pruning) and performance evaluation metrics.
        - Familiarity with RESTful API design to expose AI models as services.

        **Soft Skills:**
        - Excellent communication skills and the ability to work collaboratively in a fast-paced environment.
        - Problem-solving mindset with the ability to think critically and strategically.
        - Strong attention to detail and ability to manage multiple tasks and projects simultaneously.
        - Passion for continuous learning and adapting to new tools and technologies.

        ---

         **Nice-to-Have Skills:**
        - Familiarity with MLOps practices (CI/CD pipelines for machine learning).
        - Experience with reinforcement learning techniques and frameworks.
        - Knowledge of time-series forecasting and related algorithms.
        - Experience in applying AI/ML solutions within industries such as healthcare, finance, or logistics.
        
        ---

         **Benefits:**
        - Competitive salary and performance-based bonuses.
        - Comprehensive health, dental, and vision insurance.
        - Flexible work schedule with remote options.
        - Opportunities for professional development, training, and conferences.
        - Stock options or equity participation based on performance.

        '''}
    return context