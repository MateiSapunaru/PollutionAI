"""
Enhanced Visualization Module for Soil Contamination Dataset
Compatible with new scientific dataset structure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

class SoilVisualization:
    """
    Comprehensive visualization class for soil contamination analysis
    """
    
    def __init__(self, filepath):
        """Initialize with dataset"""
        self.df = pd.read_csv(filepath)
        self.output_dir = 'visualizations'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        print(f"✓ Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def overview_statistics(self):
        """Display dataset overview"""
        print("\n" + "="*80)
        print("DATASET OVERVIEW".center(80))
        print("="*80)
        
        print(f"\nTotal Records: {len(self.df)}")
        print(f"Date Range: {self.df['Date_Reported'].min()} to {self.df['Date_Reported'].max()}")
        print(f"\nPollutant Types: {', '.join(self.df['Pollutant_Type'].unique())}")
        print(f"Regions: {', '.join(self.df['Region'].unique())}")
        print(f"Countries: {len(self.df['Country'].unique())}")
        
        print("\n" + "-"*80)
        print("CONCENTRATION STATISTICS")
        print("-"*80)
        print(self.df[['Total_Concentration_mg_kg', 'Bioavailable_Concentration_mg_kg', 'Soil_pH']].describe())
        
        print("\n" + "-"*80)
        print("DISEASE DISTRIBUTION")
        print("-"*80)
        print(self.df['Disease_Severity'].value_counts())
    
    def ph_bioavailability_analysis(self, save=False):
        """Analyze pH effects on bioavailability"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. pH vs Bioavailability for cationic metals
        cationic = self.df[self.df['Pollutant_Type'].isin(['Lead', 'Cadmium', 'Mercury', 'Chromium'])]
        cationic['Bioavail_%'] = (cationic['Bioavailable_Concentration_mg_kg'] / 
                                   cationic['Total_Concentration_mg_kg'] * 100)
        
        for metal in ['Lead', 'Cadmium', 'Mercury', 'Chromium']:
            data = cationic[cationic['Pollutant_Type'] == metal]
            axes[0, 0].scatter(data['Soil_pH'], data['Bioavail_%'], 
                              label=metal, alpha=0.6, s=30)
        
        axes[0, 0].set_title('pH vs Bioavailability (Cationic Metals)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Soil pH')
        axes[0, 0].set_ylabel('Bioavailable (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Arsenic (anionic) - inverse relationship
        arsenic = self.df[self.df['Pollutant_Type'] == 'Arsenic'].copy()
        arsenic['Bioavail_%'] = (arsenic['Bioavailable_Concentration_mg_kg'] / 
                                 arsenic['Total_Concentration_mg_kg'] * 100)
        
        axes[0, 1].scatter(arsenic['Soil_pH'], arsenic['Bioavail_%'], 
                          color='darkred', alpha=0.6, s=30)
        axes[0, 1].set_title('pH vs Bioavailability (Arsenic - Anionic)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Soil pH')
        axes[0, 1].set_ylabel('Bioavailable (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. pH categories vs bioavailability
        ph_categories = pd.cut(self.df['Soil_pH'], bins=[4, 5.5, 7, 8.5], 
                               labels=['Acidic\n(<5.5)', 'Neutral\n(5.5-7)', 'Alkaline\n(>7)'])
        self.df['pH_Category'] = ph_categories
        
        cationic_cat = self.df[self.df['Pollutant_Type'].isin(['Lead', 'Cadmium', 'Mercury', 'Chromium'])].copy()
        cationic_cat['Bioavail_%'] = (cationic_cat['Bioavailable_Concentration_mg_kg'] / 
                                       cationic_cat['Total_Concentration_mg_kg'] * 100)
        
        cationic_cat.boxplot(column='Bioavail_%', by='pH_Category', ax=axes[1, 0])
        axes[1, 0].set_title('Bioavailability by pH Category (Cationic Metals)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('pH Category')
        axes[1, 0].set_ylabel('Bioavailable (%)')
        plt.sca(axes[1, 0])
        plt.xticks(rotation=0)
        
        # 4. Correlation heatmap
        corr_data = self.df[['Soil_pH', 'Total_Concentration_mg_kg', 
                            'Bioavailable_Concentration_mg_kg', 'Soil_Organic_Matter_%',
                            'CEC_meq_100g', 'Temperature_C']].corr()
        
        sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=axes[1, 1], square=True)
        axes[1, 1].set_title('Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.output_dir, 'ph_bioavailability_analysis.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def soil_texture_analysis(self, save=False):
        """Analyze soil texture effects"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Bioavailability by texture
        texture_bioavail = self.df.groupby('Soil_Texture').apply(
            lambda x: (x['Bioavailable_Concentration_mg_kg'] / x['Total_Concentration_mg_kg'] * 100).mean()
        ).sort_values(ascending=False)
        
        colors = ['#ff9999', '#ffcc99', '#ffff99', '#99ff99']
        axes[0, 0].bar(range(len(texture_bioavail)), texture_bioavail.values, 
                      color=colors, edgecolor='black')
        axes[0, 0].set_xticks(range(len(texture_bioavail)))
        axes[0, 0].set_xticklabels(texture_bioavail.index)
        axes[0, 0].set_title('Average Bioavailability by Soil Texture', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Bioavailable (%)')
        axes[0, 0].set_xlabel('Soil Texture')
        
        # 2. CEC distribution by texture
        self.df.boxplot(column='CEC_meq_100g', by='Soil_Texture', ax=axes[0, 1])
        axes[0, 1].set_title('CEC by Soil Texture', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Soil Texture')
        axes[0, 1].set_ylabel('CEC (meq/100g)')
        plt.sca(axes[0, 1])
        
        # 3. Disease severity by texture
        texture_severity = pd.crosstab(self.df['Soil_Texture'], self.df['Disease_Severity'])
        texture_severity.plot(kind='bar', stacked=True, ax=axes[1, 0], 
                             color=['lightgreen', 'gold', 'orange', 'red'])
        axes[1, 0].set_title('Disease Severity Distribution by Soil Texture', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Soil Texture')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend(title='Severity')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Organic matter effect
        self.df['SOM_Category'] = pd.cut(self.df['Soil_Organic_Matter_%'], 
                                         bins=[0, 2, 5, 10], 
                                         labels=['Low (<2%)', 'Medium (2-5%)', 'High (>5%)'])
        
        som_bioavail = self.df.groupby('SOM_Category').apply(
            lambda x: (x['Bioavailable_Concentration_mg_kg'] / x['Total_Concentration_mg_kg'] * 100).mean()
        )
        
        axes[1, 1].bar(range(len(som_bioavail)), som_bioavail.values, 
                      color=['red', 'orange', 'green'], edgecolor='black')
        axes[1, 1].set_xticks(range(len(som_bioavail)))
        axes[1, 1].set_xticklabels(som_bioavail.index, rotation=0)
        axes[1, 1].set_title('Bioavailability by Organic Matter Content', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Bioavailable (%)')
        axes[1, 1].set_xlabel('Soil Organic Matter')
        
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.output_dir, 'soil_texture_analysis.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def distance_decay_analysis(self, save=False):
        """Analyze distance decay patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        industrial = self.df[self.df['Nearby_Industry'] != 'None'].copy()
        
        # 1. Distance vs concentration (all industries)
        axes[0, 0].scatter(industrial['Distance_from_Source_km'], 
                          industrial['Total_Concentration_mg_kg'],
                          alpha=0.5, s=30, color='darkblue')
        axes[0, 0].set_title('Distance Decay Pattern (All Industries)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Distance from Source (km)')
        axes[0, 0].set_ylabel('Total Concentration (mg/kg)')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add exponential fit
        x = industrial['Distance_from_Source_km'].values
        y = industrial['Total_Concentration_mg_kg'].values
        mask = (x > 0) & (y > 0)
        if mask.sum() > 10:
            z = np.polyfit(x[mask], np.log(y[mask]), 1)
            p = np.poly1d(z)
            x_fit = np.linspace(x[mask].min(), x[mask].max(), 100)
            axes[0, 0].plot(x_fit, np.exp(p(x_fit)), 'r--', linewidth=2, 
                           label=f'Exponential Fit')
            axes[0, 0].legend()
        
        # 2. Distance categories
        industrial['Distance_Category'] = pd.cut(industrial['Distance_from_Source_km'],
                                                  bins=[0, 1, 3, 5, 100],
                                                  labels=['0-1 km', '1-3 km', '3-5 km', '>5 km'])
        
        dist_conc = industrial.groupby('Distance_Category')['Total_Concentration_mg_kg'].mean()
        axes[0, 1].bar(range(len(dist_conc)), dist_conc.values, 
                      color=['red', 'orange', 'yellow', 'green'], edgecolor='black')
        axes[0, 1].set_xticks(range(len(dist_conc)))
        axes[0, 1].set_xticklabels(dist_conc.index, rotation=0)
        axes[0, 1].set_title('Average Concentration by Distance', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Concentration (mg/kg)')
        axes[0, 1].set_xlabel('Distance Category')
        
        # 3. By industry type
        for industry in ['Mining', 'Battery', 'Tannery', 'Chemical']:
            ind_data = industrial[industrial['Nearby_Industry'] == industry]
            if len(ind_data) > 5:
                axes[1, 0].scatter(ind_data['Distance_from_Source_km'],
                                  ind_data['Total_Concentration_mg_kg'],
                                  label=industry, alpha=0.6, s=40)
        
        axes[1, 0].set_title('Distance Decay by Industry Type', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Distance from Source (km)')
        axes[1, 0].set_ylabel('Concentration (mg/kg)')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Disease severity vs distance
        severity_dist = industrial.groupby('Distance_Category')['Disease_Severity'].value_counts(normalize=True).unstack(fill_value=0)
        severity_dist.plot(kind='bar', stacked=True, ax=axes[1, 1],
                          color=['lightgreen', 'gold', 'orange', 'red'])
        axes[1, 1].set_title('Disease Severity by Distance', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Distance Category')
        axes[1, 1].set_ylabel('Proportion')
        axes[1, 1].legend(title='Severity')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.output_dir, 'distance_decay_analysis.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def pollutant_analysis(self, save=False):
        """Comprehensive pollutant analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Distribution by pollutant type
        pollutant_counts = self.df['Pollutant_Type'].value_counts()
        axes[0, 0].bar(pollutant_counts.index, pollutant_counts.values, 
                      color='steelblue', edgecolor='black')
        axes[0, 0].set_title('Cases by Pollutant Type', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Pollutant')
        axes[0, 0].set_ylabel('Number of Cases')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Concentration distribution
        self.df.boxplot(column='Total_Concentration_mg_kg', by='Pollutant_Type', ax=axes[0, 1])
        axes[0, 1].set_title('Concentration Distribution by Pollutant', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Pollutant Type')
        axes[0, 1].set_ylabel('Concentration (mg/kg)')
        axes[0, 1].set_yscale('log')
        plt.sca(axes[0, 1])
        plt.xticks(rotation=45)
        
        # 3. Disease types by pollutant
        disease_pollutant = pd.crosstab(self.df['Pollutant_Type'], self.df['Disease_Type'])
        top_diseases = disease_pollutant.sum().nlargest(6).index
        disease_pollutant[top_diseases].plot(kind='bar', ax=axes[1, 0], stacked=True, colormap='Set3')
        axes[1, 0].set_title('Top Disease Types by Pollutant', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Pollutant')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend(title='Disease', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Bioavailable vs Total by metal
        for pollutant in self.df['Pollutant_Type'].unique():
            data = self.df[self.df['Pollutant_Type'] == pollutant]
            axes[1, 1].scatter(data['Total_Concentration_mg_kg'],
                             data['Bioavailable_Concentration_mg_kg'],
                             label=pollutant, alpha=0.6, s=30)
        
        axes[1, 1].set_title('Bioavailable vs Total Concentration', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Total Concentration (mg/kg)')
        axes[1, 1].set_ylabel('Bioavailable Concentration (mg/kg)')
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.output_dir, 'pollutant_analysis.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def age_vulnerability_analysis(self, save=False):
        """Analyze age-specific vulnerability"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Age distribution
        age_counts = self.df['Age_Group_Affected'].value_counts()
        axes[0, 0].pie(age_counts.values, labels=age_counts.index, autopct='%1.1f%%',
                      colors=['lightblue', 'lightgreen', 'lightyellow'])
        axes[0, 0].set_title('Age Group Distribution', fontsize=14, fontweight='bold')
        
        # 2. Children vulnerability for Pb/Hg
        lead_hg = self.df[self.df['Pollutant_Type'].isin(['Lead', 'Mercury'])]
        age_lead_hg = lead_hg['Age_Group_Affected'].value_counts()
        axes[0, 1].bar(age_lead_hg.index, age_lead_hg.values, color=['red', 'orange', 'yellow'], 
                      edgecolor='black')
        axes[0, 1].set_title('Age Distribution for Lead/Mercury Cases', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Age Group')
        axes[0, 1].set_ylabel('Number of Cases')
        
        # 3. Severity by age and metal
        severity_age = pd.crosstab([self.df['Age_Group_Affected'], self.df['Pollutant_Type']], 
                                   self.df['Disease_Severity'])
        severity_age.plot(kind='bar', ax=axes[1, 0], color=['lightgreen', 'gold', 'orange', 'red'])
        axes[1, 0].set_title('Disease Severity by Age and Pollutant', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Age Group - Pollutant')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend(title='Severity')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Average bioavailable by age
        age_bioavail = self.df.groupby('Age_Group_Affected')['Bioavailable_Concentration_mg_kg'].mean()
        axes[1, 1].bar(age_bioavail.index, age_bioavail.values, 
                      color=['lightcoral', 'lightsalmon', 'gold'], edgecolor='black')
        axes[1, 1].set_title('Average Bioavailable Concentration by Age', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Age Group')
        axes[1, 1].set_ylabel('Avg Bioavailable Conc (mg/kg)')
        
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.output_dir, 'age_vulnerability_analysis.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def industry_contamination_patterns(self, save=False):
        """Analyze industry-specific patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Cases by industry
        industry_counts = self.df['Nearby_Industry'].value_counts()
        axes[0, 0].barh(industry_counts.index, industry_counts.values, color='teal', edgecolor='black')
        axes[0, 0].set_title('Cases by Industry Type', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Number of Cases')
        
        # 2. Average concentration by industry
        industry_conc = self.df.groupby('Nearby_Industry')['Total_Concentration_mg_kg'].mean().sort_values(ascending=False)
        axes[0, 1].bar(range(len(industry_conc)), industry_conc.values, 
                      color='coral', edgecolor='black')
        axes[0, 1].set_xticks(range(len(industry_conc)))
        axes[0, 1].set_xticklabels(industry_conc.index, rotation=45, ha='right')
        axes[0, 1].set_title('Average Concentration by Industry', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Concentration (mg/kg)')
        
        # 3. Primary metals by industry
        industry_metal = pd.crosstab(self.df['Nearby_Industry'], self.df['Pollutant_Type'])
        industry_metal.plot(kind='bar', stacked=True, ax=axes[1, 0], colormap='tab10')
        axes[1, 0].set_title('Metal Distribution by Industry', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Industry')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend(title='Metal', bbox_to_anchor=(1.05, 1))
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Severity by industry
        industry_severity = pd.crosstab(self.df['Nearby_Industry'], self.df['Disease_Severity'])
        industry_severity.plot(kind='bar', ax=axes[1, 1], 
                              color=['lightgreen', 'gold', 'orange', 'red'])
        axes[1, 1].set_title('Disease Severity by Industry', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Industry')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend(title='Severity')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.output_dir, 'industry_patterns.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("\n" + "="*80)
        print("GENERATING ALL VISUALIZATIONS".center(80))
        print("="*80 + "\n")
        
        self.overview_statistics()
        
        print("\n📊 Generating pH-Bioavailability Analysis...")
        self.ph_bioavailability_analysis(save=True)
        
        print("📊 Generating Soil Texture Analysis...")
        self.soil_texture_analysis(save=True)
        
        print("📊 Generating Distance Decay Analysis...")
        self.distance_decay_analysis(save=True)
        
        print("📊 Generating Pollutant Analysis...")
        self.pollutant_analysis(save=True)
        
        print("📊 Generating Age Vulnerability Analysis...")
        self.age_vulnerability_analysis(save=True)
        
        print("📊 Generating Industry Pattern Analysis...")
        self.industry_contamination_patterns(save=True)
        
        print("\n" + "="*80)
        print(f"✓ ALL VISUALIZATIONS SAVED TO '{self.output_dir}/'".center(80))
        print("="*80 + "\n")


def main():
    """Main visualization function"""
    filepath = 'data/soil_contamination_scientific.csv'
    
    if not os.path.exists(filepath):
        print(f"❌ Error: Dataset not found at {filepath}")
        return
    
    viz = SoilVisualization(filepath)
    viz.generate_all_visualizations()


if __name__ == "__main__":
    main()