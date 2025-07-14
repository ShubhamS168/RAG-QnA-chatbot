import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

class DataLoader:
    """Handles data loading and preprocessing for the RAG system"""
    
    def __init__(self, config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_loan_data(self) -> pd.DataFrame:
        """Load loan approval dataset"""
        # Sample loan data (in production, load from actual CSV)
        sample_data = {
            'Loan_ID': ['LP001001', 'LP001002', 'LP001003', 'LP001004', 'LP001005'],
            'Gender': ['Male', 'Male', 'Male', 'Male', 'Female'],
            'Married': ['No', 'No', 'Yes', 'Yes', 'No'],
            'Dependents': ['0', '1', '0', '0', '0'],
            'Education': ['Graduate', 'Graduate', 'Graduate', 'Not Graduate', 'Graduate'],
            'Self_Employed': ['No', 'No', 'No', 'No', 'No'],
            'ApplicantIncome': [5849, 4583, 3000, 2583, 6000],
            'CoapplicantIncome': [0, 1508, 0, 2358, 0],
            'LoanAmount': [146, 128, 66, 120, 141],
            'Loan_Amount_Term': [360, 360, 360, 360, 360],
            'Credit_History': [1, 1, 1, 1, 1],
            'Property_Area': ['Urban', 'Rural', 'Urban', 'Urban', 'Urban'],
            'Loan_Status': ['Y', 'N', 'Y', 'Y', 'Y']
        }
        
        df = pd.DataFrame(sample_data)
        return df
    
    def create_documents_from_data(self, df: pd.DataFrame) -> List[Document]:
        """Convert dataset into searchable documents"""
        documents = []
        
        # Create dataset overview document
        overview_text = f"""
        Loan Approval Prediction Dataset Overview:
        - Total Records: {len(df)}
        - Features: {len(df.columns)}
        - Target Variable: Loan_Status (Y/N)
        - Approval Rate: {(df['Loan_Status'] == 'Y').mean():.2%}
        
        Key Features:
        - Demographic: Gender, Married, Dependents, Education
        - Financial: ApplicantIncome, CoapplicantIncome, LoanAmount
        - Credit: Credit_History, Loan_Amount_Term
        - Property: Property_Area
        """
        
        documents.append(Document(
            page_content=overview_text,
            metadata={"source": "dataset_overview", "type": "summary"}
        ))
        
        # Create feature-specific documents
        for column in df.columns:
            if column != 'Loan_ID':
                feature_analysis = self._analyze_feature(df, column)
                documents.append(Document(
                    page_content=feature_analysis,
                    metadata={"source": f"feature_{column}", "type": "analysis"}
                ))
        
        # Create sample records documents
        for idx, row in df.head(10).iterrows():
            record_text = f"""
            Loan Application Record:
            ID: {row['Loan_ID']}
            Applicant Profile:
            - Gender: {row['Gender']}
            - Married: {row['Married']}
            - Dependents: {row['Dependents']}
            - Education: {row['Education']}
            - Self Employed: {row['Self_Employed']}
            
            Financial Information:
            - Applicant Income: ${row['ApplicantIncome']:,}
            - Coapplicant Income: ${row['CoapplicantIncome']:,}
            - Loan Amount: ${row['LoanAmount']:,}
            - Loan Term: {row['Loan_Amount_Term']} months
            - Credit History: {row['Credit_History']}
            
            Property and Decision:
            - Property Area: {row['Property_Area']}
            - Loan Status: {'Approved' if row['Loan_Status'] == 'Y' else 'Rejected'}
            """
            
            documents.append(Document(
                page_content=record_text,
                metadata={"source": f"record_{row['Loan_ID']}", "type": "record"}
            ))
        
        return documents
    
    def _analyze_feature(self, df: pd.DataFrame, column: str) -> str:
        """Analyze individual feature statistics"""
        if df[column].dtype in ['int64', 'float64']:
            stats = df[column].describe()
            analysis = f"""
            Feature Analysis: {column}
            Type: Numerical
            Statistics:
            - Mean: {stats['mean']:.2f}
            - Median: {stats['50%']:.2f}
            - Min: {stats['min']:.2f}
            - Max: {stats['max']:.2f}
            - Standard Deviation: {stats['std']:.2f}
            
            Loan Approval Impact:
            - Average for Approved: {df[df['Loan_Status'] == 'Y'][column].mean():.2f}
            - Average for Rejected: {df[df['Loan_Status'] == 'N'][column].mean():.2f}
            """
        else:
            value_counts = df[column].value_counts()
            analysis = f"""
            Feature Analysis: {column}
            Type: Categorical
            Categories: {list(value_counts.index)}
            Distribution: {dict(value_counts)}
            
            Loan Approval by Category:
            """
            for category in value_counts.index:
                subset = df[df[column] == category]
                approval_rate = (subset['Loan_Status'] == 'Y').mean()
                analysis += f"- {category}: {approval_rate:.2%} approval rate\n"
        
        return analysis
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks for better retrieval"""
        return self.text_splitter.split_documents(documents)
    
    def get_dataset_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive dataset statistics"""
        stats = {
            "total_records": len(df),
            "features": len(df.columns),
            "approval_rate": (df['Loan_Status'] == 'Y').mean(),
            "average_loan_amount": df['LoanAmount'].mean(),
            "average_income": df['ApplicantIncome'].mean(),
            "feature_types": {
                col: str(df[col].dtype) for col in df.columns
            }
        }
        return stats
