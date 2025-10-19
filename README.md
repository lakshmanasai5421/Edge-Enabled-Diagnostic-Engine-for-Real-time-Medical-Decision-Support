# Edge-Enabled Diagnostic Engine for Real-time Medical Decision Support

This project introduces a secure, web-based diagnostic engine designed to provide real-time medical decision support while rigorously protecting patient confidentiality. At its core, the system uses a custom-built lightweight homomorphic encryption scheme (`LPME.py`) to ensure that sensitive patient data remains encrypted, even during the machine learning model training process.

This platform allows for the secure analysis of medical data to diagnose conditions like heart disease and hypothyroidism, providing a critical tool for healthcare professionals without ever compromising the privacy of the individuals they serve.

##  Our Vision

In an era where data is invaluable, medical data is sacred. The challenge is to harness the power of machine learning for diagnostics without exposing sensitive patient information. This project was born from that challenge. We envision a future where advanced AI can assist doctors in real-time, anywhere in the world, without forcing a trade-off between innovation and privacy. This tool is a step toward that future, building a foundation of trust between technology and healthcare.

##  Key Features

  * **Peace of Mind with Privacy-by-Design:** We don't just protect data; we build on it. Our system trains advanced machine learning models (like Naive Bayes and XGBoost) directly on *encrypted* data. Sensitive patient details are never exposed, not even to the system administrators.
  * **Custom-Built Lightweight Encryption:** Powered by our `LPME.py` module, the system uses a lightweight homomorphic encryption scheme. This allows the server to perform computations (like model training) on data it cannot read, ensuring end-to-end confidentiality.
  * **Instant Diagnostic Support:** Get immediate, AI-powered insights. Upload a new patient's data (as a simple CSV file), and the system provides a real-time prediction (e.g., "Heart Disease Detected") based on the privacy-preserving model.
  * **User-Friendly Web Interface:** A clean and intuitive Flask application guides users through every step, from secure login and data encryption to model training and final diagnosis.
  * **Intelligent Data Handling:** The system is smart. It automatically handles common data science challenges, such as using SMOTE to correct for imbalanced datasets (like in the hypothyroid data), leading to more reliable and accurate models.

##  Technology Stack

  * **Backend:** Python, Flask
  * **Machine Learning:** Scikit-learn, XGBoost, Pandas, NumPy
  * **Data Balancing:** Imbalanced-learn (SMOTE)
  * **Privacy/Encryption:** Custom `LPME.py` module
  * **Model/Data Storage:** Joblib, Pickle

##  How It Works: A Step-by-Step Journey

The application's architecture is built on a clear, security-first workflow:

1.  **Step 1: Secure Access:** A user first creates an account or logs in through the secure authentication portal.
2.  **Step 2: The "Digital Safe" - Data Encryption:** The administrator selects a raw medical dataset (e.g., `heart.csv`). The `LPME.py` module then "locks" this data, encrypting each feature individually and saving it as a new, unreadable file (e.g., `enc_heart.csv`).
3.  **Step 3: Training the "Blind" Expert:** The admin initiates model training. The system uses the *encrypted* dataset to train the machine learning models. This is the magic: the models learn to find patterns and make predictions without ever "seeing" the original, sensitive information.
4.  **Step 4: Real-time, Secure Diagnosis:** A doctor or user uploads a new, unencrypted test file. The system instantly processes this new data, loads the "blind-trained" model, and provides a clear, actionable diagnosis (e.g., "Thyroid detected"), all while maintaining the integrity of the privacy-preserving workflow.

##  Getting Started: Setup and Installation

Follow these steps to get the project running on your local machine.

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Set Up a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    We recommend creating a `requirements.txt` file with the following content:

    ```txt
    Flask
    scikit-learn
    pandas
    numpy
    xgboost
    imbalanced-learn
    joblib
    ```

    Then, install them:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Create the Necessary Folders:**
    For the app to work correctly, create these directories in your project's root:

    ```
    /
    ├── Dataset/
    │   ├── heart.csv
    │   └── hypothyroid.csv
    ├── EncryptedData/
    ├── Model/
    ├── static/
    ├── templates/
    └── main.py
    ```

      * Place your raw `.csv` files in the `Dataset/` folder.
      * Your HTML files (`index.html`, `Login.html`, etc.) go in the `templates/` folder.

5.  **Run the Application:**

    ```bash
    python main.py
    ```

    Open your browser and navigate to `http://127.0.0.1:8080`.

## Application Workflow

1.  Visit `http://127.0.0.1:8080`.
2.  **Sign up** for a new account, then **Login**.
3.  From the **Admin Screen**, follow these steps:
      * **Encrypt Data:** Click **"Select Dataset to Encrypt"**, choose a dataset, and submit. You'll see a comparison of the plain and encrypted data.
      * **Train Models:** Click **"Click to Train ML Models"**. The system will train on the encrypted data and show you a performance report.
      * **Get Predictions:** Click **"Select Dataset to Test"**, choose the model type, and upload your test CSV file. You'll receive instant diagnostic results.
\! Whether you're looking to fix a bug, improve the encryption module, or add a new feature, please feel free to fork the repository and submit a pull request.
