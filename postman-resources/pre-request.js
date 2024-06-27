let userMessage = `Here is user rating history:
* Title: Jenna Jameson Heartbreaker Perfume for women 3.4 oz Eau De Parfum Spray, Rating: 1.0 stars
* Title: OZNaturals Anti Aging Retinol Serum -The Most Effective Anti Wrinkle Serum Contains Professional Strength Retinol+ Astaxanthin+ Vitamin E - Get The Dramatic Youthful Results You&rsquo;ve Been Looking For, Rating: 4.0 stars
* Title: Kordon Oasis (Novalek) Bell Bottle 8oz, Rating: 5.0 stars

Here is the rating history from users who are similar to this user:
* Title: Philips Norelco HQ110 Shaving Head Cleaning Spray, Rating: 5.0 stars
* Title: Citre Shine Moisture Burst Shampoo - 16 fl oz, Rating: 4.0 stars
* Title: Yardley By Yardley Of London Unisexs Lay It On Thick Hand &amp; Foot Cream 5.3 Oz, Rating: 5.0 stars
* Title: &quot;BAD ASS&quot; Masculine Pheromone Cologne with the &quot;ADRENALINE&quot; Fragrance From SpellboundRX - The Intelligent Pheromone Choice for Intelligent People. GUARANTEED!, Rating: 5.0 stars


Based on above rating history and similar users' rating history, please predict user's rating for the product Title: Caboodles Heart Throb Long Tapered Tote, Black Diamond, 1.12 Pound, (1 being lowest and 5 being highest,The output should be like: (x stars, xx%), do not explain the reason.)`;


userMessage = userMessage.replace(/"/g, '\\"');
userMessage = userMessage.replace(/\n/g, '\\n');
userMessage = userMessage.replace(/\t/g, '\\t');


pm.variables.set('userMessage', userMessage);
