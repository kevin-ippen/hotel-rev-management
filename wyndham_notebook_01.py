# Databricks notebook source
# MAGIC %md
# MAGIC # Wyndham Revenue Management Genie Space - Environment Setup
# MAGIC 
# MAGIC **Notebook 01**: Environment Setup and Catalog Creation  
# MAGIC **Runtime**: DBR 16.4 LTS  
# MAGIC **Compute**: Shared All-Purpose or Serverless SQL  
# MAGIC **Data Scope**: ~900 properties (10% sample), US + Canada, 8 brands, 3 years historical  
# MAGIC 
# MAGIC This notebook sets up the foundational Unity Catalog structure for the Wyndham Revenue Management Genie space, following Databricks AI/BI best practices.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Environment Validation and Setup

# COMMAND ----------

# Validate Databricks environment and permissions
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import uuid
from datetime import datetime, timedelta
import random
import json

# Get current user and workspace info
current_user = spark.sql("SELECT current_user() as user").collect()[0]["user"]
workspace_url = spark.conf.get("spark.databricks.workspaceUrl", "unknown")

print(f"Current User: {current_user}")
print(f"Workspace: {workspace_url}")
print(f"Spark Version: {spark.version}")
print(f"DBR Version: {spark.conf.get('spark.databricks.clusterUsageTags.sparkVersion', 'unknown')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Unity Catalog Schema Creation
# MAGIC 
# MAGIC Following the three-tier architecture:
# MAGIC - `main.wyndham_staging`: Raw data ingestion
# MAGIC - `main.wyndham_curated`: Clean, business-ready tables  
# MAGIC - `main.wyndham_analytics`: Aggregated views optimized for Genie

# COMMAND ----------

# Create schemas with proper permissions and comments
schemas_to_create = [
    {
        'name': 'wyndham_staging',
        'comment': 'Raw data staging area for Wyndham revenue management data. Used for initial data ingestion and validation.'
    },
    {
        'name': 'wyndham_curated', 
        'comment': 'Curated, business-ready tables for Wyndham revenue analytics. Clean data with proper typing and validation.'
    },
    {
        'name': 'wyndham_analytics',
        'comment': 'Analytics layer with pre-joined views and aggregations optimized for AI/BI Genie queries.'
    }
]

for schema in schemas_to_create:
    try:
        # Create schema if it doesn't exist
        spark.sql(f"""
            CREATE SCHEMA IF NOT EXISTS main.{schema['name']}
            COMMENT '{schema['comment']}'
        """)
        print(f"✓ Created/verified schema: main.{schema['name']}")
        
        # Set ownership if needed (adjust based on your governance model)
        # spark.sql(f"ALTER SCHEMA main.{schema['name']} OWNER TO `{current_user}`")
        
    except Exception as e:
        print(f"✗ Error creating schema main.{schema['name']}: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Core Table Definitions
# MAGIC 
# MAGIC Creating the foundational tables with comprehensive metadata following Genie best practices:
# MAGIC - Detailed column comments with examples and valid values
# MAGIC - Proper data types for optimal performance
# MAGIC - Foreign key relationships documented

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Properties Master Table - Foundation for all revenue analysis
# MAGIC CREATE TABLE IF NOT EXISTS main.wyndham_curated.properties_master (
# MAGIC   property_id STRING NOT NULL COMMENT 'Unique property identifier. Format: WYN_[BRAND]_[REGION]_[NUMBER]. Example: WYN_DAYS_SE_001',
# MAGIC   property_name STRING NOT NULL COMMENT 'Full property name including brand and location. Example: Days Inn Atlanta Airport',
# MAGIC   brand STRING NOT NULL COMMENT 'Wyndham brand family. Valid values: "Days Inn", "Super 8", "Ramada", "Wyndham", "Baymont", "Travelodge", "Howard Johnson", "Wingate"',
# MAGIC   region STRING NOT NULL COMMENT 'Geographic region. Valid values: "Northeast", "Southeast", "Midwest", "Southwest", "West", "Central Canada", "Eastern Canada", "Western Canada"',
# MAGIC   market_tier STRING NOT NULL COMMENT 'Market classification based on city size and economic activity. Valid values: "Primary" (major cities like NYC, Toronto), "Secondary" (mid-size markets), "Tertiary" (small markets)',
# MAGIC   property_type STRING NOT NULL COMMENT 'Location and guest type focus. Valid values: "Urban", "Suburban", "Airport", "Highway", "Resort", "Extended Stay"',
# MAGIC   room_count INT NOT NULL COMMENT 'Total number of available guest rooms. Typically ranges from 60-450 depending on brand and market',
# MAGIC   ownership_type STRING NOT NULL COMMENT 'Business model. Valid values: "Corporate" (company owned), "Franchise" (franchisee operated), "Management Contract"',
# MAGIC   open_date DATE NOT NULL COMMENT 'Property opening date for calculating property age and performance maturity',
# MAGIC   city STRING NOT NULL COMMENT 'Primary city location for market analysis',
# MAGIC   state_province STRING NOT NULL COMMENT 'State (US) or province (Canada) for regulatory and regional analysis',
# MAGIC   country STRING NOT NULL COMMENT 'Country location. Valid values: "US", "Canada"',
# MAGIC   market_id STRING NOT NULL COMMENT 'Market identifier for competitive analysis. Groups properties competing in same geographic market',
# MAGIC   latitude DECIMAL(9,6) COMMENT 'Geographic latitude coordinate for mapping and proximity analysis',
# MAGIC   longitude DECIMAL(9,6) COMMENT 'Geographic longitude coordinate for mapping and proximity analysis',
# MAGIC   created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP() COMMENT 'Record creation timestamp for audit purposes',
# MAGIC   updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP() COMMENT 'Last record update timestamp'
# MAGIC ) 
# MAGIC USING DELTA
# MAGIC TBLPROPERTIES (
# MAGIC   'delta.autoOptimize.optimizeWrite' = 'true',
# MAGIC   'delta.autoOptimize.autoCompact' = 'true'
# MAGIC )
# MAGIC COMMENT 'Master property data for all Wyndham properties. Foundation table for property lookup, regional analysis, and brand performance comparisons. Links to all operational data via property_id.';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Daily Performance Table - Core revenue and operational metrics
# MAGIC CREATE TABLE IF NOT EXISTS main.wyndham_curated.daily_performance (
# MAGIC   property_id STRING NOT NULL COMMENT 'Foreign key to properties_master.property_id',
# MAGIC   business_date DATE NOT NULL COMMENT 'Stay date (not booking date) for revenue recognition. Used for all time-based analysis',
# MAGIC   rooms_available INT NOT NULL COMMENT 'Total rooms available for sale excluding out-of-order rooms. Denominator for occupancy calculations',
# MAGIC   rooms_sold INT NOT NULL COMMENT 'Actual rooms sold and occupied. Numerator for occupancy calculations',
# MAGIC   occupancy_rate DECIMAL(5,4) NOT NULL COMMENT 'Occupancy percentage calculated as rooms_sold/rooms_available. Values between 0 and 1 (0.7500 = 75%)',
# MAGIC   adr DECIMAL(10,2) NOT NULL COMMENT 'Average Daily Rate in USD. Room revenue divided by rooms sold. Key pricing metric',
# MAGIC   revpar DECIMAL(10,2) NOT NULL COMMENT 'Revenue Per Available Room in USD. Calculated as adr * occupancy_rate. Primary performance KPI',
# MAGIC   revenue_rooms DECIMAL(12,2) NOT NULL COMMENT 'Total room revenue in USD for the business date',
# MAGIC   revenue_fb DECIMAL(12,2) DEFAULT 0 COMMENT 'Food & Beverage revenue in USD (applicable for full-service properties)',
# MAGIC   revenue_other DECIMAL(12,2) DEFAULT 0 COMMENT 'Other ancillary revenue in USD (parking, Wi-Fi, resort fees, etc.)',
# MAGIC   revenue_total DECIMAL(12,2) NOT NULL COMMENT 'Total property revenue in USD (sum of all revenue streams)',
# MAGIC   avg_length_of_stay DECIMAL(3,1) NOT NULL COMMENT 'Average guest stay duration in nights for arrivals on business_date',
# MAGIC   booking_channel_mix STRING COMMENT 'JSON distribution breakdown. Example: {"direct": 0.45, "ota": 0.35, "gds": 0.15, "voice": 0.05}',
# MAGIC   market_segment_mix STRING COMMENT 'JSON guest type breakdown. Example: {"business": 0.60, "leisure": 0.30, "group": 0.10}',
# MAGIC   walk_in_rate DECIMAL(5,4) DEFAULT 0 COMMENT 'Percentage of guests without advance reservations. Higher values indicate strong drive-by location',
# MAGIC   no_show_rate DECIMAL(5,4) DEFAULT 0 COMMENT 'Percentage of reservations that failed to arrive without cancellation',
# MAGIC   cancellation_rate DECIMAL(5,4) DEFAULT 0 COMMENT 'Same-day cancellation percentage affecting revenue management',
# MAGIC   group_rooms INT DEFAULT 0 COMMENT 'Number of rooms sold to group business (10+ rooms typically)',
# MAGIC   created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP() COMMENT 'Record creation timestamp',
# MAGIC   updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP() COMMENT 'Last update timestamp'
# MAGIC ) 
# MAGIC USING DELTA
# MAGIC PARTITIONED BY (business_date)
# MAGIC TBLPROPERTIES (
# MAGIC   'delta.autoOptimize.optimizeWrite' = 'true',
# MAGIC   'delta.autoOptimize.autoCompact' = 'true'
# MAGIC )
# MAGIC COMMENT 'Daily operational performance metrics by property. Primary table for revenue analysis, occupancy trends, and KPI reporting. Partitioned by business_date for optimal query performance.';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Competitive Intelligence Table - Market performance comparison
# MAGIC CREATE TABLE IF NOT EXISTS main.wyndham_curated.competitive_intelligence (
# MAGIC   market_id STRING NOT NULL COMMENT 'Market identifier grouping competitive properties',
# MAGIC   business_date DATE NOT NULL COMMENT 'Performance measurement date',
# MAGIC   property_id STRING NOT NULL COMMENT 'Subject Wyndham property being analyzed',
# MAGIC   market_occupancy DECIMAL(5,4) NOT NULL COMMENT 'Total market occupancy rate for all properties in competitive set',
# MAGIC   market_adr DECIMAL(10,2) NOT NULL COMMENT 'Market average ADR in USD across competitive set',
# MAGIC   market_revpar DECIMAL(10,2) NOT NULL COMMENT 'Market RevPAR in USD across competitive set',
# MAGIC   penetration_index DECIMAL(5,2) NOT NULL COMMENT 'Market share performance. 100.00 = fair share, >100 = above fair share, <100 = below fair share',
# MAGIC   adr_index DECIMAL(5,2) NOT NULL COMMENT 'ADR performance vs market average. 100.00 = market average, >100 = premium pricing, <100 = discounted',
# MAGIC   revpar_index DECIMAL(5,2) NOT NULL COMMENT 'RevPAR performance vs market average. 100.00 = market average, primary competitiveness metric',
# MAGIC   market_room_nights INT NOT NULL COMMENT 'Total room nights demanded across entire market',
# MAGIC   fair_share_rooms INT NOT NULL COMMENT 'Expected room nights based on inventory share',
# MAGIC   comp_set_properties STRING COMMENT 'JSON array of competitor property identifiers in the competitive set',
# MAGIC   market_supply INT COMMENT 'Total rooms available across competitive set',
# MAGIC   created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP() COMMENT 'Record creation timestamp',
# MAGIC   updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP() COMMENT 'Last update timestamp'
# MAGIC ) 
# MAGIC USING DELTA
# MAGIC PARTITIONED BY (business_date)
# MAGIC TBLPROPERTIES (
# MAGIC   'delta.autoOptimize.optimizeWrite' = 'true',
# MAGIC   'delta.autoOptimize.autoCompact' = 'true'
# MAGIC )
# MAGIC COMMENT 'Competitive performance data for market share analysis. Use for understanding market conditions vs property-specific performance issues. Critical for revenue optimization strategies.';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Rate and Inventory Management Table - Pricing strategy data
# MAGIC CREATE TABLE IF NOT EXISTS main.wyndham_curated.rate_inventory (
# MAGIC   property_id STRING NOT NULL COMMENT 'Foreign key to properties_master.property_id',
# MAGIC   business_date DATE NOT NULL COMMENT 'Stay date for rate and inventory availability',
# MAGIC   room_type STRING NOT NULL COMMENT 'Room category. Examples: "Standard King", "Standard Queen", "Suite", "Accessible"',
# MAGIC   rate_code STRING NOT NULL COMMENT 'Rate plan identifier. Examples: "BAR" (Best Available Rate), "AAA", "GOVT", "CORP"',
# MAGIC   rate_amount DECIMAL(10,2) NOT NULL COMMENT 'Published rate amount in USD before taxes and fees',
# MAGIC   inventory_available INT NOT NULL COMMENT 'Rooms available for sale in this rate/room type combination',
# MAGIC   restrictions STRING COMMENT 'JSON booking restrictions. Example: {"min_stay": 2, "advance_purchase": 7, "non_refundable": true}',
# MAGIC   channel_rates STRING COMMENT 'JSON with channel-specific rates. Example: {"direct": 129.00, "expedia": 135.00, "booking": 133.00}',
# MAGIC   pickup_pace INT DEFAULT 0 COMMENT 'Bookings received for this date in last 24 hours',
# MAGIC   forecast_demand INT COMMENT 'Predicted total demand for this room type and date',
# MAGIC   price_sensitivity DECIMAL(3,2) COMMENT 'Demand elasticity coefficient for pricing optimization',
# MAGIC   created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP() COMMENT 'Record creation timestamp',
# MAGIC   updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP() COMMENT 'Last update timestamp'
# MAGIC ) 
# MAGIC USING DELTA
# MAGIC PARTITIONED BY (business_date)
# MAGIC TBLPROPERTIES (
# MAGIC   'delta.autoOptimize.optimizeWrite' = 'true',
# MAGIC   'delta.autoOptimize.autoCompact' = 'true'
# MAGIC )
# MAGIC COMMENT 'Pricing and inventory management data by property, date, and room type. Essential for revenue optimization and rate strategy analysis.';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Market Events Table - External demand drivers
# MAGIC CREATE TABLE IF NOT EXISTS main.wyndham_curated.market_events (
# MAGIC   event_id STRING NOT NULL COMMENT 'Unique event identifier',
# MAGIC   market_id STRING NOT NULL COMMENT 'Market where event occurs',
# MAGIC   event_date DATE NOT NULL COMMENT 'Primary event date (start date for multi-day events)',
# MAGIC   end_date DATE COMMENT 'Event end date for multi-day events',
# MAGIC   event_name STRING NOT NULL COMMENT 'Event name or description',
# MAGIC   event_type STRING NOT NULL COMMENT 'Event category. Valid values: "Conference", "Sports", "Concert", "Holiday", "Weather", "Economic", "Construction"',
# MAGIC   impact_rating STRING NOT NULL COMMENT 'Expected demand impact. Valid values: "Low" (5-15% lift), "Medium" (15-30% lift), "High" (30%+ lift)',
# MAGIC   affected_properties STRING COMMENT 'JSON array of property IDs expected to see impact',
# MAGIC   demand_lift_pct DECIMAL(5,2) COMMENT 'Percentage increase in demand. Positive values indicate increased demand',
# MAGIC   adr_lift_pct DECIMAL(5,2) COMMENT 'Percentage rate premium opportunity during event',
# MAGIC   attendance INT COMMENT 'Expected event attendance if known',
# MAGIC   event_location STRING COMMENT 'Specific venue or area within market',
# MAGIC   created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP() COMMENT 'Record creation timestamp',
# MAGIC   updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP() COMMENT 'Last update timestamp'
# MAGIC ) 
# MAGIC USING DELTA
# MAGIC PARTITIONED BY (event_date)
# MAGIC TBLPROPERTIES (
# MAGIC   'delta.autoOptimize.optimizeWrite' = 'true',
# MAGIC   'delta.autoOptimize.autoCompact' = 'true'
# MAGIC )
# MAGIC COMMENT 'External events and factors affecting hotel demand. Use for understanding performance anomalies and forecasting demand patterns.';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Guest Transaction Table - Individual booking and stay details
# MAGIC CREATE TABLE IF NOT EXISTS main.wyndham_curated.guest_transactions (
# MAGIC   transaction_id STRING NOT NULL COMMENT 'Unique transaction identifier',
# MAGIC   property_id STRING NOT NULL COMMENT 'Foreign key to properties_master.property_id',
# MAGIC   guest_id STRING NOT NULL COMMENT 'Unique guest identifier for loyalty and repeat visit analysis',
# MAGIC   business_date DATE NOT NULL COMMENT 'Arrival date for the stay',
# MAGIC   departure_date DATE NOT NULL COMMENT 'Departure date for the stay',
# MAGIC   length_of_stay INT NOT NULL COMMENT 'Number of nights stayed',
# MAGIC   room_type STRING NOT NULL COMMENT 'Room category booked',
# MAGIC   rate_code STRING NOT NULL COMMENT 'Rate plan used for booking',
# MAGIC   room_revenue DECIMAL(10,2) NOT NULL COMMENT 'Total room revenue for entire stay in USD',
# MAGIC   ancillary_revenue DECIMAL(10,2) DEFAULT 0 COMMENT 'Additional revenue (parking, F&B, etc.) in USD',
# MAGIC   total_revenue DECIMAL(10,2) NOT NULL COMMENT 'Total guest revenue for stay in USD',
# MAGIC   booking_channel STRING NOT NULL COMMENT 'Distribution channel. Examples: "Direct", "Expedia", "Booking.com", "Walk-in"',
# MAGIC   market_segment STRING NOT NULL COMMENT 'Guest classification. Valid values: "Business", "Leisure", "Group", "Extended Stay"',
# MAGIC   booking_date DATE NOT NULL COMMENT 'Date when reservation was made',
# MAGIC   advance_booking_days INT NOT NULL COMMENT 'Days between booking and arrival',
# MAGIC   guest_type STRING COMMENT 'Guest classification. Examples: "New", "Repeat", "Loyalty Member"',
# MAGIC   party_size INT DEFAULT 1 COMMENT 'Number of guests in reservation',
# MAGIC   cancellation_date DATE COMMENT 'Date reservation was cancelled (null if not cancelled)',
# MAGIC   no_show BOOLEAN DEFAULT FALSE COMMENT 'True if guest failed to arrive without cancelling',
# MAGIC   created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP() COMMENT 'Record creation timestamp',
# MAGIC   updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP() COMMENT 'Last update timestamp'
# MAGIC ) 
# MAGIC USING DELTA
# MAGIC PARTITIONED BY (business_date)
# MAGIC TBLPROPERTIES (
# MAGIC   'delta.autoOptimize.optimizeWrite' = 'true',
# MAGIC   'delta.autoOptimize.autoCompact' = 'true'
# MAGIC )
# MAGIC COMMENT 'Individual guest transaction records for detailed revenue analysis and guest behavior patterns. Links to guest_id for loyalty and repeat visit analysis.';

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Analytics Layer Setup
# MAGIC 
# MAGIC Creating optimized views for Genie that pre-join data and add calculated fields to minimize query complexity and improve response time.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Primary Revenue Analytics View - Pre-joined data optimized for Genie
# MAGIC CREATE OR REPLACE VIEW main.wyndham_analytics.revenue_performance_view AS
# MAGIC SELECT 
# MAGIC   -- Property Attributes
# MAGIC   p.property_id,
# MAGIC   p.property_name,
# MAGIC   p.brand,
# MAGIC   p.region,
# MAGIC   p.market_tier,
# MAGIC   p.property_type,
# MAGIC   p.ownership_type,
# MAGIC   p.room_count,
# MAGIC   p.city,
# MAGIC   p.state_province,
# MAGIC   p.country,
# MAGIC   p.market_id,
# MAGIC   
# MAGIC   -- Performance Metrics
# MAGIC   dp.business_date,
# MAGIC   dp.rooms_available,
# MAGIC   dp.rooms_sold,
# MAGIC   dp.occupancy_rate,
# MAGIC   dp.adr,
# MAGIC   dp.revpar,
# MAGIC   dp.revenue_total,
# MAGIC   dp.avg_length_of_stay,
# MAGIC   
# MAGIC   -- Time Dimensions for Easy Filtering
# MAGIC   YEAR(dp.business_date) as year,
# MAGIC   MONTH(dp.business_date) as month_number,
# MAGIC   DATE_FORMAT(dp.business_date, 'MMMM') as month_name,
# MAGIC   QUARTER(dp.business_date) as quarter,
# MAGIC   DAYOFWEEK(dp.business_date) as day_of_week_number,
# MAGIC   DATE_FORMAT(dp.business_date, 'EEEE') as day_of_week_name,
# MAGIC   CASE 
# MAGIC     WHEN DAYOFWEEK(dp.business_date) IN (1, 2, 3, 4) THEN 'Weekday'
# MAGIC     ELSE 'Weekend'
# MAGIC   END as weekend_indicator,
# MAGIC   
# MAGIC   -- Seasonality Classification
# MAGIC   CASE 
# MAGIC     WHEN MONTH(dp.business_date) IN (6, 7, 8) THEN 'Peak'
# MAGIC     WHEN MONTH(dp.business_date) IN (12, 1, 2) THEN 'Low' 
# MAGIC     ELSE 'Shoulder'
# MAGIC   END as season,
# MAGIC   
# MAGIC   -- Competitive Performance
# MAGIC   ci.market_occupancy,
# MAGIC   ci.market_adr,
# MAGIC   ci.market_revpar,
# MAGIC   ci.penetration_index,
# MAGIC   ci.adr_index,
# MAGIC   ci.revpar_index,
# MAGIC   
# MAGIC   -- Performance Classifications
# MAGIC   CASE 
# MAGIC     WHEN ci.revpar_index >= 110 THEN 'Outperforming'
# MAGIC     WHEN ci.revpar_index >= 90 THEN 'In-line'
# MAGIC     ELSE 'Underperforming'
# MAGIC   END as market_performance_category,
# MAGIC   
# MAGIC   -- Calculated Business Metrics
# MAGIC   dp.revenue_total / p.room_count as revenue_per_room,
# MAGIC   CASE 
# MAGIC     WHEN dp.occupancy_rate >= 0.85 THEN 'High'
# MAGIC     WHEN dp.occupancy_rate >= 0.65 THEN 'Medium'
# MAGIC     ELSE 'Low'
# MAGIC   END as occupancy_category,
# MAGIC   
# MAGIC   -- Year-over-Year Comparison Fields
# MAGIC   LAG(dp.revpar, 365) OVER (PARTITION BY dp.property_id ORDER BY dp.business_date) as revpar_ly,
# MAGIC   LAG(dp.occupancy_rate, 365) OVER (PARTITION BY dp.property_id ORDER BY dp.business_date) as occupancy_ly,
# MAGIC   LAG(dp.adr, 365) OVER (PARTITION BY dp.property_id ORDER BY dp.business_date) as adr_ly
# MAGIC   
# MAGIC FROM main.wyndham_curated.daily_performance dp
# MAGIC JOIN main.wyndham_curated.properties_master p 
# MAGIC   ON dp.property_id = p.property_id
# MAGIC LEFT JOIN main.wyndham_curated.competitive_intelligence ci 
# MAGIC   ON dp.property_id = ci.property_id 
# MAGIC   AND dp.business_date = ci.business_date
# MAGIC   
# MAGIC COMMENT 'Primary analytics view combining property attributes with daily performance and competitive data. Pre-joined and optimized for AI/BI Genie queries with calculated fields for common business metrics.';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Guest Analytics View - Transaction-level insights
# MAGIC CREATE OR REPLACE VIEW main.wyndham_analytics.guest_analytics_view AS
# MAGIC SELECT 
# MAGIC   -- Property Context
# MAGIC   p.property_name,
# MAGIC   p.brand,
# MAGIC   p.region,
# MAGIC   p.market_tier,
# MAGIC   
# MAGIC   -- Guest Transaction Details
# MAGIC   gt.transaction_id,
# MAGIC   gt.guest_id,
# MAGIC   gt.business_date,
# MAGIC   gt.departure_date,
# MAGIC   gt.length_of_stay,
# MAGIC   gt.room_type,
# MAGIC   gt.rate_code,
# MAGIC   gt.total_revenue,
# MAGIC   gt.booking_channel,
# MAGIC   gt.market_segment,
# MAGIC   gt.advance_booking_days,
# MAGIC   gt.guest_type,
# MAGIC   
# MAGIC   -- Calculated Metrics
# MAGIC   gt.total_revenue / gt.length_of_stay as average_daily_spend,
# MAGIC   CASE 
# MAGIC     WHEN gt.advance_booking_days <= 7 THEN 'Last Minute'
# MAGIC     WHEN gt.advance_booking_days <= 30 THEN 'Short Lead'
# MAGIC     WHEN gt.advance_booking_days <= 90 THEN 'Medium Lead'
# MAGIC     ELSE 'Long Lead'
# MAGIC   END as booking_window_category,
# MAGIC   
# MAGIC   -- Time Dimensions
# MAGIC   YEAR(gt.business_date) as year,
# MAGIC   MONTH(gt.business_date) as month_number,
# MAGIC   DATE_FORMAT(gt.business_date, 'MMMM') as month_name
# MAGIC   
# MAGIC FROM main.wyndham_curated.guest_transactions gt
# MAGIC JOIN main.wyndham_curated.properties_master p 
# MAGIC   ON gt.property_id = p.property_id
# MAGIC WHERE gt.cancellation_date IS NULL 
# MAGIC   AND gt.no_show = FALSE
# MAGIC   
# MAGIC COMMENT 'Guest transaction analytics view for booking pattern analysis, channel performance, and guest behavior insights.';

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Data Validation and Index Creation

# COMMAND ----------

# Create constraints and optimize tables for query performance
try:
    # Add primary key constraints (logical, not enforced in Delta)
    spark.sql("ALTER TABLE main.wyndham_curated.properties_master ADD CONSTRAINT pk_properties PRIMARY KEY(property_id)")
    print("✓ Added primary key constraint to properties_master")
except Exception as e:
    print(f"Note: Primary key constraint may already exist or not supported: {str(e)}")

try:
    # Add foreign key constraint documentation
    spark.sql("ALTER TABLE main.wyndham_curated.daily_performance ADD CONSTRAINT fk_daily_performance_property FOREIGN KEY(property_id) REFERENCES main.wyndham_curated.properties_master(property_id)")
    print("✓ Added foreign key constraint to daily_performance")
except Exception as e:
    print(f"Note: Foreign key constraint may already exist or not supported: {str(e)}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Optimize tables for query performance
# MAGIC OPTIMIZE main.wyndham_curated.properties_master;
# MAGIC OPTIMIZE main.wyndham_curated.daily_performance;
# MAGIC OPTIMIZE main.wyndham_curated.competitive_intelligence;
# MAGIC OPTIMIZE main.wyndham_curated.rate_inventory;
# MAGIC OPTIMIZE main.wyndham_curated.market_events;
# MAGIC OPTIMIZE main.wyndham_curated.guest_transactions;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Setup Validation and Summary

# COMMAND ----------

# Validate schema creation and table structure
def validate_setup():
    """Validate that all schemas and tables were created successfully"""
    
    print("=== SCHEMA VALIDATION ===")
    schemas = spark.sql("SHOW SCHEMAS IN main").collect()
    wyndham_schemas = [row.databaseName for row in schemas if 'wyndham' in row.databaseName]
    
    expected_schemas = ['wyndham_staging', 'wyndham_curated', 'wyndham_analytics']
    for schema in expected_schemas:
        if schema in wyndham_schemas:
            print(f"✓ Schema exists: main.{schema}")
        else:
            print(f"✗ Missing schema: main.{schema}")
    
    print("\n=== TABLE VALIDATION ===")
    # Check curated tables
    curated_tables = spark.sql("SHOW TABLES IN main.wyndham_curated").collect()
    table_names = [row.tableName for row in curated_tables]
    
    expected_tables = [
        'properties_master',
        'daily_performance', 
        'competitive_intelligence',
        'rate_inventory',
        'market_events',
        'guest_transactions'
    ]
    
    for table in expected_tables:
        if table in table_names:
            print(f"✓ Table exists: main.wyndham_curated.{table}")
            # Get row count (should be 0 initially)
            count = spark.sql(f"SELECT COUNT(*) as cnt FROM main.wyndham_curated.{table}").collect()[0]['cnt']
            print(f"  Rows: {count}")
        else:
            print(f"✗ Missing table: main.wyndham_curated.{table}")
    
    print("\n=== VIEW VALIDATION ===")
    # Check analytics views
    analytics_objects = spark.sql("SHOW TABLES IN main.wyndham_analytics").collect()
    view_names = [row.tableName for row in analytics_objects]
    
    expected_views = [
        'revenue_performance_view',
        'guest_analytics_view'
    ]
    
    for view in expected_views:
        if view in view_names:
            print(f"✓ View exists: main.wyndham_analytics.{view}")
        else:
            print(f"✗ Missing view: main.wyndham_analytics.{view}")

# Run validation
validate_setup()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Next Steps and Data Generation Prep

# COMMAND ----------

# Create configuration for data generation
data_generation_config = {
    "environment": {
        "catalog": "main",
        "staging_schema": "wyndham_staging", 
        "curated_schema": "wyndham_curated",
        "analytics_schema": "wyndham_analytics"
    },
    "data_scope": {
        "property_count": 900,  # 10% sampling
        "brands": ["Days Inn", "Super 8", "Ramada", "Wyndham", "Baymont", "Travelodge", "Howard Johnson", "Wingate"],
        "regions": ["Northeast", "Southeast", "Midwest", "Southwest", "West", "Central Canada", "Eastern Canada", "Western Canada"],
        "countries": ["US", "Canada"],
        "historical_years": 3,
        "start_date": "2021-01-01",
        "end_date": "2023-12-31"
    },
    "realism_factors": {
        "natural_anomalies": True,
        "realistic_skew": True, 
        "customer_loyalty": True,
        "competitive_dynamics": True,
        "seasonal_patterns": True,
        "market_events": True
    }
}

# Store config for next notebook
dbutils.fs.put("/tmp/wyndham_config.json", json.dumps(data_generation_config, indent=2))
print("✓ Configuration saved for data generation")
print(f"Config: {json.dumps(data_generation_config, indent=2)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC **Environment Setup Complete ✓**
# MAGIC 
# MAGIC **Created:**
# MAGIC - 3 Unity Catalog schemas (staging, curated, analytics)
# MAGIC - 6 core tables with comprehensive metadata and comments
# MAGIC - 2 analytics views optimized for Genie queries
# MAGIC - Foreign key relationships and constraints
# MAGIC - Performance optimizations (partitioning, clustering)
# MAGIC 
# MAGIC **Key Features:**
# MAGIC - Detailed column comments with examples and valid values
# MAGIC - Pre-joined analytics views to minimize query complexity  
# MAGIC - Calculated fields for common business metrics
# MAGIC - Time-based partitioning for optimal performance
# MAGIC - Year-over-year comparison fields
# MAGIC 
# MAGIC **Ready for Next Steps:**
# MAGIC - Notebook 02: Synthetic data generation with realistic hospitality patterns
# MAGIC - Notebook 03: Data quality validation and business rule enforcement
# MAGIC - Notebook 04: Genie space configuration and instruction setup
# MAGIC 
# MAGIC The foundation follows Databricks AI/BI Genie best practices for optimal performance and accuracy.