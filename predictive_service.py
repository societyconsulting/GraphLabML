import graphlab as gl
import time


if __name__ == '__main__':
    # More info can be found in the GraphLab documentation and on the GraphLab
    # forums at graphlab.com/products/create/docs and forum.graphlab.com

    # Note: You must run analyze_algorithms.py first to generate the logistic
    # regression model (model.log).

    # Note: I would recommend running this in an IPython session. This code
    # sometimes executes too fast before the predictive service can be set up
    # properly, hence the need for a time.sleep(10).

    # Set up the ec2 environment. Note you should have  AWS_ACCESS_KEY_ID and
    # AWS_SECRET_ACCESS_KEY set in your .bash_profile
    s3_bucket = 's3://society-gl-demo'
    ec2_env = gl.deploy.environment.EC2('bot_det_env', s3_bucket)

    # Load the model.
    model_log = gl.load_model('model.log')

    # Start the predicitive services. WARNING: This will cost money so don't
    # run this code if you don't want to pay for the EC2 instances. This will
    # set up 3 m3.xlarge instances.
    s3_path = s3_bucket + '/predictive-services'
    ps = gl.deploy.predictive_service.create('bot-pred-serv', ec2_env, s3_path)
    ps.add('log-classifier', model_log)
    ps.apply_changes()

    # Wait awhile so our predictive service can be fully loaded.
    time.sleep(10)

    # You can view the GraphLab canvas by calling.
    ps.show()

    # You can view your predictive services by examining.
    #gl.deploy.predictive_services

    # If you exit your session and want to grab your predictive service object
    # again you can do so by running.
    #ps = gl.deploy.predictive_services[0]
    
    # Query our predictive service.
    ps.query('log-classifier', {'method': 'predict', 'data': {'dataset': 
        {'event_counts': 1000, 'std': 55.5}}})

    # Call this when you want to terminate the service or terminate the service
    # manually from the AWS console.
    #ps.terminate_service()

    # You can also query the REST API using the following curl command. You can
    # find your API Key and DNS Name by running ps.show()
    # curl -X POST -d '{"api key": "<Your API Key>", "data": {"method": 
    # "predict", "data": {"event_counts": 1000, "std": 55.5}}}' 
    # http://<AWS DNS Name>/data/log-classifier
