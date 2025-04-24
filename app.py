import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from tensorflow.keras.models import load_model

# Custom CSS for professional styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Set page config with professional theme
st.set_page_config(
    page_title="Food Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
local_css("styles.css")  # We'll create this file below

# Load data and model (same as before)
@st.cache_data
def load_data():
    center_info = pd.read_csv('fulfilment_center_info.csv')
    meal_info = pd.read_csv('meal_info.csv')
    train_data = pd.read_csv('genpact_train.csv')
    test_data = pd.read_csv('genpact_test.csv')
    
    df = pd.merge(pd.merge(train_data, center_info, how='inner', on='center_id'), 
                 meal_info, how='inner', on='meal_id')
    test = pd.merge(pd.merge(test_data, center_info, how='inner', on='center_id'), 
                   meal_info, how='inner', on='meal_id')
    
    cat_var = ['center_type', 'cuisine', 'category']
    for i in cat_var:
        df[i] = pd.factorize(df[i])[0]
        test[i] = pd.factorize(test[i])[0]
    
    return df, test

@st.cache_resource
def load_bilstm_model():
    return load_model('bilstm_model.h5')

df, test = load_data()
model = load_bilstm_model()

# Category mapping
category_mapping = {
    0: "Beverages",
    1: "Biryani",
    2: "Desert",
    3: "Extras",
    4: "Fish",
    5: "Other Snacks",
    6: "Pasta",
    7: "Pizza",
    8: "Rice Bowl",
    9: "Salad",
    10: "Sandwich",
    11: "Seafood",
    12: "Soup",
    13: "Starters"
}

# Main app with professional styling
def main():
    # Custom header with gradient
    st.markdown("""
    <div class="header">
        <h1>Food Demand Forecasting System</h1>
        <p>Advanced predictive analytics for food service operations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with professional styling
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2>Navigation</h2>
        </div>
        """, unsafe_allow_html=True)
        
        app_mode = st.radio(
            "Menu",
            ["Dashboard", "Data Exploration", "Forecast Analysis", "About"],
            key="nav"
        )
        
        st.markdown("---")

    if app_mode == "Dashboard":
        # Professional dashboard layout
        st.markdown("## 📊 Operational Overview")
        
        # Key metrics row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Total Historical Orders</h3>
                <p>{:,}</p>
            </div>
            """.format(df['num_orders'].sum()), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Unique Meal Items</h3>
                <p>{}</p>
            </div>
            """.format(df['meal_id'].nunique()), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>Fulfillment Centers</h3>
                <p>{}</p>
            </div>
            """.format(df['center_id'].nunique()), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Main chart
        st.markdown("### Weekly Demand Trend")
        weekly_demand = df.groupby('week')['num_orders'].sum().reset_index()
        fig = px.line(weekly_demand, x='week', y='num_orders',
                      template="plotly_white",
                      color_discrete_sequence=["#4C78A8"])
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Week",
            yaxis_title="Number of Orders"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif app_mode == "Data Exploration":
        # Professional data exploration layout
        st.markdown("## 🔍 Data Exploration")
        
        with st.expander("📁 Data Preview", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Training Data**")
                st.dataframe(df.head())
            with col2:
                st.markdown("**Test Data**")
                st.dataframe(test.head())
        
        st.markdown("---")
        
        # Interactive visualizations
        st.markdown("### 📈 Interactive Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Demand Trends", "Category Analysis", "Center Performance"])
        
        with tab1:
            st.markdown("#### Weekly Demand Patterns")
            weekly_demand = df.groupby('week')['num_orders'].sum().reset_index()
            fig = px.line(weekly_demand, x='week', y='num_orders',
                          template="plotly_white",
                          color_discrete_sequence=["#4C78A8"])
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("#### Category Distribution")
            category_demand = df.copy()
            category_demand['category'] = category_demand['category'].map(category_mapping)
            category_demand = category_demand.groupby('category')['num_orders'].sum().reset_index()
            fig = px.bar(category_demand, x='category', y='num_orders',
                         template="plotly_white",
                         color_discrete_sequence=["#72B7B2"])
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("#### Center Type Performance")
            center_demand = df.groupby('center_type')['num_orders'].sum().reset_index()
            fig = px.pie(center_demand, values='num_orders', names='center_type',
                         template="plotly_white",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)
    
    elif app_mode == "Forecast Analysis":
        # Professional forecast analysis layout
        st.markdown("## 🔮 Forecast Analysis")
        
        with st.expander("⚙️ Filter Options", expanded=True):
            selected_categories = st.multiselect(
                "Select Categories",
                options=list(category_mapping.values()),
                default=["Beverages", "Biryani"]
            )
            
            forecast_button = st.button(
                "Generate Forecast",
                type="primary",
                use_container_width=True
            )
        
        if forecast_button or selected_categories:
            with st.spinner("Generating forecast..."):
                # Reverse mapping for categories
                rev_category_mapping = {v: k for k, v in category_mapping.items()}
                selected_cat_codes = [rev_category_mapping[cat] for cat in selected_categories]
                
                # Filter both historical and test data by selected categories
                historical = df[df['category'].isin(selected_cat_codes)]
                filtered_test = test[test['category'].isin(selected_cat_codes)]
                
                if not filtered_test.empty and not historical.empty:
                    # Make predictions
                    test_pred = filtered_test.values.astype(np.float32)
                    test_reshaped = test_pred.reshape((test_pred.shape[0], 7, test_pred.shape[1] // 7))
                    predictions = model.predict(test_reshaped)
                    filtered_test['predicted_orders'] = predictions
                    
                    # =============================================
                    # FIRST: Show TOTAL forecast across all categories
                    # =============================================
                    st.markdown("## 📊 Total Forecast (All Selected Categories)")
                    
                    # Aggregate historical data
                    total_hist_agg = historical.groupby('week')['num_orders'].sum().reset_index()
                    total_hist_agg['type'] = 'Historical'
                    
                    # Aggregate predicted data
                    total_pred_agg = filtered_test.groupby('week')['predicted_orders'].sum().reset_index()
                    total_pred_agg = total_pred_agg.rename(columns={'predicted_orders': 'num_orders'})
                    total_pred_agg['type'] = 'Forecast'
                    
                    # Combine for plotting
                    total_combined = pd.concat([total_hist_agg, total_pred_agg])
                    
                    # Create total forecast plot
                    fig_total = px.line(
                        total_combined, 
                        x='week', 
                        y='num_orders', 
                        color='type',
                        template="plotly_white",
                        color_discrete_map={
                            'Historical': '#4C78A8',
                            'Forecast': '#E45756'
                        },
                        title="Combined Demand Forecast"
                    )
                    
                    fig_total.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis_title="Week",
                        yaxis_title="Number of Orders",
                        legend_title="Data Type",
                        hovermode="x unified"
                    )
                    
                    # Add forecast separation line
                    max_train_week_total = total_hist_agg['week'].max()
                    fig_total.add_vline(
                        x=max_train_week_total + 0.5,
                        line_width=2,
                        line_dash="dash",
                        line_color="#72B7B2",
                        annotation_text="Forecast Start",
                        annotation_position="top left"
                    )
                    
                    st.plotly_chart(fig_total, use_container_width=True)
                    
                    # Add total metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Total Historical Orders", 
                            f"{total_hist_agg['num_orders'].sum():,}"
                        )
                    
                    with col2:
                        st.metric(
                            "Total Forecasted Orders", 
                            f"{total_pred_agg['num_orders'].sum():,}"
                        )
                    
                    with col3:
                        total_growth = ((total_pred_agg['num_orders'].sum() - total_hist_agg['num_orders'].sum()) / 
                                    total_hist_agg['num_orders'].sum()) * 100
                        st.metric(
                            "Overall Demand Change",
                            f"{total_growth:.1f}%",
                            delta_color="inverse"
                        )
                    
                    st.markdown("---")
                    
                    # =============================================
                    # SECOND: Show individual category forecasts
                    # =============================================
                    st.markdown("## 📈 Category-Specific Forecasts")
                    
                    # Create a separate chart for each selected category
                    for category in selected_categories:
                        cat_code = rev_category_mapping[category]
                        
                        # Prepare data for current category
                        hist_cat = historical[historical['category'] == cat_code]
                        test_cat = filtered_test[filtered_test['category'] == cat_code]
                        
                        hist_agg = hist_cat.groupby('week')['num_orders'].sum().reset_index()
                        hist_agg['type'] = 'Historical'
                        
                        pred_agg = test_cat.groupby('week')['predicted_orders'].sum().reset_index()
                        pred_agg = pred_agg.rename(columns={'predicted_orders': 'num_orders'})
                        pred_agg['type'] = 'Forecast'
                        
                        combined = pd.concat([hist_agg, pred_agg])
                        
                        # Create plot for current category
                        st.markdown(f"### {category} Demand")
                        
                        fig = px.line(
                            combined, 
                            x='week', 
                            y='num_orders', 
                            color='type',
                            template="plotly_white",
                            color_discrete_map={
                                'Historical': '#4C78A8',
                                'Forecast': '#E45756'
                            }
                        )
                        
                        fig.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            xaxis_title="Week",
                            yaxis_title="Number of Orders",
                            legend_title="Data Type",
                            hovermode="x unified"
                        )
                        
                        # Add forecast separation line
                        max_train_week = hist_agg['week'].max()
                        fig.add_vline(
                            x=max_train_week + 0.5,
                            line_width=2,
                            line_dash="dash",
                            line_color="#72B7B2",
                            annotation_text="Forecast Start",
                            annotation_position="top left"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add metrics for each category
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                f"Historical Orders", 
                                f"{hist_agg['num_orders'].sum():,}"
                            )
                        
                        with col2:
                            st.metric(
                                f"Forecasted Orders", 
                                f"{pred_agg['num_orders'].sum():,}"
                            )
                        
                        st.markdown("---")
                    
                    # Show raw data toggle
                    with st.expander("📋 View Forecast Data", expanded=False):
                        display_df = filtered_test.copy()
                        display_df['category'] = display_df['category'].map(category_mapping)
                        st.dataframe(display_df[['week', 'center_id', 'meal_id', 'category', 'predicted_orders']])
                
                else:
                    st.warning("No data matches your selected categories")
    elif app_mode == "About":
        # Professional about page
        st.markdown("## ℹ️ About This Project")
        
        st.markdown("""
        <div class="about-container">
            <h3>Food Demand Forecasting System</h3>
            <p>This application provides advanced predictive analytics for food service operations using state-of-the-art deep learning techniques.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Technical Architecture Card
        st.markdown("""
        <div class="feature-card">
            <h4>🔧 Technical Architecture</h4>
            <ul>
                <li><strong>Model:</strong> Bidirectional LSTM Neural Network</li>
                <li><strong>Framework:</strong> TensorFlow/Keras</li>
                <li><strong>Visualization:</strong> Plotly Interactive Charts</li>
                <li><strong>Interface:</strong> Streamlit Dashboard</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Features Card
        st.markdown("""
        <div class="feature-card">
            <h4>💡 Key Features</h4>
            <ul>
                <li>Interactive demand forecasting</li>
                <li>Multi-dimensional filtering</li>
                <li>Historical vs forecast comparison</li>
                <li>Performance metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Business Value Card
        st.markdown("""
        <div class="feature-card">
            <h4>📈 Business Value</h4>
            <ul>
                <li>Optimize inventory management</li>
                <li>Improve staff scheduling</li>
                <li>Reduce food waste</li>
                <li>Enhance promotional planning</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional styling if needed
        st.markdown("""
        <style>
            .feature-card {
                background: white;
                border-radius: 0.5rem;
                padding: 1.5rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                margin-bottom: 1.5rem;
            }
            .feature-card h4 {
                color: #4C78A8;
                margin-top: 0;
                border-bottom: 1px solid #eee;
                padding-bottom: 0.5rem;
            }
            .feature-card ul {
                padding-left: 1.5rem;
            }
            .feature-card li {
                margin-bottom: 0.5rem;
            }
            .about-container {
                max-width: 900px;
                margin: 0 auto 2rem auto;
            }
        </style>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()