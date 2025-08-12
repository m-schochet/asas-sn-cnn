from pyasassn.client import SkyPatrolClient
# The variable `save_path` is the location on your device for where to save the quiescent light curves. 
# Insert your own respective save path here to store the light curves

# Note: there are—as expected from the "Million Quasar Catalog"—about one million quasar light curves stored 
# in this catalog at the ASAS-SN Servers. 
# Since the quantity of files is quite large, ensure that you have at least 200GB of available storage 
# in the location where you intend to download these files


if __name__ == "__main__":
    save_path = '(# insert your path here #)'
    
    #ADQL query
    query= """
    SELECT *
    FROM milliquas 
    """
    client = SkyPatrolClient()

    client.adql_query(query, download=True, save_dir=save_path, file_format='parquet', threads=32)