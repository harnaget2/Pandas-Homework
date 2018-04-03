

```python
import json
import pandas as pd
from sklearn import preprocessing
```


```python
data=json.load(open('HeroesOfPymoli\purchase_data.json'))
heroes_df=pd.DataFrame(data)
heroes_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Item ID</th>
      <th>Item Name</th>
      <th>Price</th>
      <th>SN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>38</td>
      <td>Male</td>
      <td>165</td>
      <td>Bone Crushing Silver Skewer</td>
      <td>3.37</td>
      <td>Aelalis34</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
      <td>Male</td>
      <td>119</td>
      <td>Stormbringer, Dark Blade of Ending Misery</td>
      <td>2.32</td>
      <td>Eolo46</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34</td>
      <td>Male</td>
      <td>174</td>
      <td>Primitive Blade</td>
      <td>2.46</td>
      <td>Assastnya25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>Male</td>
      <td>92</td>
      <td>Final Critic</td>
      <td>1.36</td>
      <td>Pheusrical25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>23</td>
      <td>Male</td>
      <td>63</td>
      <td>Stormfury Mace</td>
      <td>1.27</td>
      <td>Aela59</td>
    </tr>
  </tbody>
</table>
</div>




```python
players_count=heroes_df.groupby(["SN"]).count()
total_players=heroes_df["SN"].unique()
total_players_count = len(total_players)
new_heroes_df = heroes_df.drop_duplicates("SN")
new_heroes_df
data_for_totals = {'Total Players': [total_players_count]}
total_players=pd.DataFrame(data_for_totals)
total_players.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total Players</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>573</td>
    </tr>
  </tbody>
</table>
</div>




```python
unique_items=heroes_df["Item ID"].unique()
average_price=heroes_df["Price"].mean()
num_purchases=heroes_df["SN"].count()
total_revenue=heroes_df["Price"].sum()

data_for_purchases = {'Number of Unique Items': [len(unique_items)], 'Average Price': [average_price], 'Number of Purchases': [num_purchases], 'Total Revenue': [total_revenue]}
table_for_purchases=pd.DataFrame(data_for_purchases)
table_for_purchases["Average Price"] = table_for_purchases["Average Price"].map("${:.2f}".format)
table_for_purchases["Total Revenue"] = table_for_purchases["Total Revenue"].map("${:.2f}".format)
table_for_purchases
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Average Price</th>
      <th>Number of Purchases</th>
      <th>Number of Unique Items</th>
      <th>Total Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>$2.93</td>
      <td>780</td>
      <td>183</td>
      <td>$2286.33</td>
    </tr>
  </tbody>
</table>
</div>




```python
gender_df_group=heroes_df.groupby("SN")

gender_df=pd.DataFrame(new_heroes_df["Gender"].value_counts())
gender_df["Percentage of Players"] = gender_df["Gender"] / total_players_count * 100
gender_df.columns =["Total Count", "Percentage of Players"]
gender_df.round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total Count</th>
      <th>Percentage of Players</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Male</th>
      <td>465</td>
      <td>81.15</td>
    </tr>
    <tr>
      <th>Female</th>
      <td>100</td>
      <td>17.45</td>
    </tr>
    <tr>
      <th>Other / Non-Disclosed</th>
      <td>8</td>
      <td>1.40</td>
    </tr>
  </tbody>
</table>
</div>




```python
gender_analysis = heroes_df[["Gender", "Price"]]
gender_analysis_df=pd.DataFrame(heroes_df["Gender"].value_counts())
gender_analysis_df.columns=["Total Count"]
gender_group = gender_analysis.groupby(["Gender"])

gender_table = gender_group.sum()
gender_table.columns=["Total Purchase Value"]
gender_table["Purchase Count"] = gender_analysis_df["Total Count"]
gender_table["Average Purchase Price"] = (gender_table["Total Purchase Value"]/gender_table["Purchase Count"]).map("${:.2f}".format)
normalizer = gender_table[["Total Purchase Value"]].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(normalizer)

gender_table["Normalized Totals"] = x_scaled
gender_table["Total Purchase Value"] = gender_table["Total Purchase Value"].map("${:.2f}".format)
gender_table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total Purchase Value</th>
      <th>Purchase Count</th>
      <th>Average Purchase Price</th>
      <th>Normalized Totals</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female</th>
      <td>$382.91</td>
      <td>136</td>
      <td>$2.82</td>
      <td>0.189509</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>$1867.68</td>
      <td>633</td>
      <td>$2.95</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Other / Non-Disclosed</th>
      <td>$35.74</td>
      <td>11</td>
      <td>$3.25</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
bins=[0, 10, 14, 19, 24, 29, 34, 39, 40]
names=[">10", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", ">40"]

heroes_df["Age Group"]=pd.cut(heroes_df["Age"], bins, labels=names)
age_groups=heroes_df.groupby("Age Group")

age_sums = age_groups["Price"].sum()

age_table=age_groups.count()
age_analysis=pd.DataFrame(age_table["Age"])
age_analysis.columns = ["Total Count"]
age_analysis["Percentage of Players"] = (age_analysis["Total Count"]/total_players_count*100).round(2)
age_analysis["Average Purchase Price"] = (age_sums/age_analysis["Total Count"]).map("${:.2f}".format)
age_analysis["Total Purchase Value"] = age_sums
age_analysis = age_analysis.rename(columns={"Total Count": "Purchase Count"}) 

normalizer = age_analysis[["Total Purchase Value"]].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(normalizer)

age_analysis["Normalized Totals"] = x_scaled
age_analysis["Total Purchase Value"]=age_analysis["Total Purchase Value"].map("${:.2f}".format)

age_analysis
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Purchase Count</th>
      <th>Percentage of Players</th>
      <th>Average Purchase Price</th>
      <th>Total Purchase Value</th>
      <th>Normalized Totals</th>
    </tr>
    <tr>
      <th>Age Group</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>&gt;10</th>
      <td>32</td>
      <td>5.58</td>
      <td>$3.02</td>
      <td>$96.62</td>
      <td>0.055170</td>
    </tr>
    <tr>
      <th>10-14</th>
      <td>31</td>
      <td>5.41</td>
      <td>$2.70</td>
      <td>$83.79</td>
      <td>0.041428</td>
    </tr>
    <tr>
      <th>15-19</th>
      <td>133</td>
      <td>23.21</td>
      <td>$2.91</td>
      <td>$386.42</td>
      <td>0.365561</td>
    </tr>
    <tr>
      <th>20-24</th>
      <td>336</td>
      <td>58.64</td>
      <td>$2.91</td>
      <td>$978.77</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25-29</th>
      <td>125</td>
      <td>21.82</td>
      <td>$2.96</td>
      <td>$370.33</td>
      <td>0.348328</td>
    </tr>
    <tr>
      <th>30-34</th>
      <td>64</td>
      <td>11.17</td>
      <td>$3.08</td>
      <td>$197.25</td>
      <td>0.162950</td>
    </tr>
    <tr>
      <th>35-39</th>
      <td>42</td>
      <td>7.33</td>
      <td>$2.84</td>
      <td>$119.40</td>
      <td>0.079569</td>
    </tr>
    <tr>
      <th>&gt;40</th>
      <td>14</td>
      <td>2.44</td>
      <td>$3.22</td>
      <td>$45.11</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
highest_spenders_group = heroes_df.groupby("SN")
highest_df = pd.DataFrame(highest_spenders_group["Price"].sum())
highest_df.columns = ["Total Purchase Amount"]

purchase_amount = highest_spenders_group["Price"].count()
purchase_amount

highest_df["Purchase Count"] = purchase_amount
highest_df["Average Purchase Price"] = (highest_df["Total Purchase Amount"]/highest_df["Purchase Count"]).map("${:.2f}".format)

spenders = highest_df.nlargest(5, "Total Purchase Amount")

spenders["Total Purchase Amount"] = spenders["Total Purchase Amount"].map("${:.2f}".format)
spenders
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total Purchase Amount</th>
      <th>Purchase Count</th>
      <th>Average Purchase Price</th>
    </tr>
    <tr>
      <th>SN</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Undirrala66</th>
      <td>$17.06</td>
      <td>5</td>
      <td>$3.41</td>
    </tr>
    <tr>
      <th>Saedue76</th>
      <td>$13.56</td>
      <td>4</td>
      <td>$3.39</td>
    </tr>
    <tr>
      <th>Mindimnya67</th>
      <td>$12.74</td>
      <td>4</td>
      <td>$3.18</td>
    </tr>
    <tr>
      <th>Haellysu29</th>
      <td>$12.73</td>
      <td>3</td>
      <td>$4.24</td>
    </tr>
    <tr>
      <th>Eoda93</th>
      <td>$11.58</td>
      <td>3</td>
      <td>$3.86</td>
    </tr>
  </tbody>
</table>
</div>




```python
most_pop_item_group=heroes_df.groupby(["Item ID", "Item Name"])
most_pop_item = pd.DataFrame(most_pop_item_group["Price"].sum())
most_pop_item.columns = ["Total Purchase Amount"]


num_purchased = most_pop_item_group["Price"].count()
most_pop_item["Purchase Count"] = num_purchased
most_pop_item["Item Price"] = (most_pop_item_group["Price"].sum()/num_purchased).map("${:.2f}".format)
                               
best_items = most_pop_item.nlargest(5, "Purchase Count")
best_items["Total Purchase Amount"] = best_items["Total Purchase Amount"].map("${:.2f}".format)
best_items
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Total Purchase Amount</th>
      <th>Purchase Count</th>
      <th>Item Price</th>
    </tr>
    <tr>
      <th>Item ID</th>
      <th>Item Name</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>39</th>
      <th>Betrayal, Whisper of Grieving Widows</th>
      <td>$25.85</td>
      <td>11</td>
      <td>$2.35</td>
    </tr>
    <tr>
      <th>84</th>
      <th>Arcane Gem</th>
      <td>$24.53</td>
      <td>11</td>
      <td>$2.23</td>
    </tr>
    <tr>
      <th>13</th>
      <th>Serenity</th>
      <td>$13.41</td>
      <td>9</td>
      <td>$1.49</td>
    </tr>
    <tr>
      <th>31</th>
      <th>Trickster</th>
      <td>$18.63</td>
      <td>9</td>
      <td>$2.07</td>
    </tr>
    <tr>
      <th>34</th>
      <th>Retribution Axe</th>
      <td>$37.26</td>
      <td>9</td>
      <td>$4.14</td>
    </tr>
  </tbody>
</table>
</div>




```python
most_profit=most_pop_item.nlargest(5, "Total Purchase Amount")
most_profit["Total Purchase Amount"] = most_profit["Total Purchase Amount"].map("${:.2f}".format)
most_profit
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Total Purchase Amount</th>
      <th>Purchase Count</th>
      <th>Item Price</th>
    </tr>
    <tr>
      <th>Item ID</th>
      <th>Item Name</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34</th>
      <th>Retribution Axe</th>
      <td>$37.26</td>
      <td>9</td>
      <td>$4.14</td>
    </tr>
    <tr>
      <th>115</th>
      <th>Spectral Diamond Doomblade</th>
      <td>$29.75</td>
      <td>7</td>
      <td>$4.25</td>
    </tr>
    <tr>
      <th>32</th>
      <th>Orenmir</th>
      <td>$29.70</td>
      <td>6</td>
      <td>$4.95</td>
    </tr>
    <tr>
      <th>103</th>
      <th>Singed Scalpel</th>
      <td>$29.22</td>
      <td>6</td>
      <td>$4.87</td>
    </tr>
    <tr>
      <th>107</th>
      <th>Splitter, Foe Of Subtlety</th>
      <td>$28.88</td>
      <td>8</td>
      <td>$3.61</td>
    </tr>
  </tbody>
</table>
</div>


