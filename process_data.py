import numpy as np
import graphlab as gl


def process_data():
    """
    This function takes our raw web data containing the client_ip, 
    server_utc_datetime (the time of the click event), bot (whether or not
    the IP was a known bot) and then produces our features which includes
    client_ip, event_counts, std and bot.
    """

    # Load our raw data into an SFrame.
    sf = gl.SFrame.read_csv('data/labeled_data.csv', header=True)

    # The first feature we want to calculate is event_counts. To do this we
    # have to group by the 'client_ip' and then sum all of the events 
    # associated with each 'client_ip'.
    sf_counts = sf.groupby('client_ip', 
            {'event_counts': gl.aggregate.COUNT('server_utc_datetime')})

    # We only want to look at IP's that have more than 50 clicks. This is
    # because it's difficult to predict whether or not an IP is a bot if we
    # only have 10 or 20 click events from that IP.
    sf_counts = sf_counts[sf_counts['event_counts'] >= 50]

    # Filter our original SFrame by relavant IP's contained in sf_counts.
    sf = sf.filter_by(sf_counts['client_ip'], 'client_ip')

    # Now we're going to caluclate 'std'.

    # Group the server_utc_datetime by the client_ip. The variable datetimes
    # is a map from client_ip to a list of server_utc_datetimes. For each
    # client_ip, we're going to sort the server_utc_datetimes, then calculate
    # the differences between each server_utc_datetime and then find the
    # standard deviation of those differences.
    datetimes = {}
    for row in sf:
        client_ip = row['client_ip']
        if client_ip not in datetimes:
            datetimes[client_ip] = [row['server_utc_datetime']]
        else:
            datetimes[client_ip].append(row['server_utc_datetime'])

    # Here we're sorting the server_utc_datetimes for each IP and then
    # calculating the standard deviation of the differences.
    stds = {}
    for ip in datetimes.keys():
        events = sorted(datetimes[ip]) 
        stds[ip] = np.std(np.diff(events))

    sf_stdv = gl.SFrame({'client_ip': stds.keys(), 'std': stds.values()})

    # Data is messy and unfortunately some of our IPs have been classified as
    # both bots and users so we need to remove those from our dataset.
    good_ips = set()
    sf.remove_column('server_utc_datetime')
    tmp = sf.unique()
    for row in tmp:
        # If we've seen the IP before, then it has been classified twice, else
        # we add it to our set of good IPs.
        client_ip = row['client_ip']
        if client_ip in good_ips:
            good_ips.remove(client_ip)
        else:
            good_ips.add(client_ip)

    tmp = tmp.filter_by(list(good_ips), 'client_ip')

    sf = sf_counts.join(sf_stdv, on='client_ip').join(tmp, on='client_ip')

    # Double check to make sure your final SFrame makes sense.
    print sf.head(10)

    # Save the dataset as features.csv. There should be a features.csv in the
    # data folder already.
    sf.save('data/features.csv', format='csv')


if __name__ == '__main__':
    process_data()
