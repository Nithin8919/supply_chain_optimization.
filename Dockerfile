# Base image for PostgreSQL
FROM postgres:latest

# Copy the data file and initialization script to the container
COPY supply_chain_data.xlsx /docker-entrypoint-initdb.d/
COPY init.sql /docker-entrypoint-initdb.d/

# Expose PostgreSQL port
EXPOSE 5432
