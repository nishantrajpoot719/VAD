from groq import Groq
import json
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Groq API key
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_food_recommendations(vad_score, intent_selections, contextual_data):
    """
    Get food recommendations from Groq API based on VAD score, intent selections, and contextual data.
    
    Args:
        vad_score: List of three values [valence, arousal, dominance]
        intent_selections: List of user preferences (e.g., ["Hot", "Light", "Tangy"])
        contextual_data: List containing [time, date, location, weather]
        
    Returns:
        Dictionary containing emotion, top products, and top combos
    """
    try:
        # Initialize Groq client with API key
        client = Groq(api_key=GROQ_API_KEY)
        
        # Create the input data in the expected format
        input_data = [vad_score, intent_selections, contextual_data]
        
        # Call the Groq API
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "system",
                    "content": """This system contains Input, Product dictionary you have to infer from input and recommend top three products and top three product combos from the product dictionary only which a person should consume at that particular time.

Input: [VAD, Intent, Contextual Data], where VAD is a list containing Valence, Arousal, and Dominance values respectively for the user's emotion (VAD values varies from -1 to 1 not 0 to 1 so there might be cases where all three values are positive that doesn't mean they are in scale of 0 to 1 they are always in scale of -1 to +1), VAD score should be used to figure out what type of food the user should consume depending on the valence, arousal, and dominance data. Intent is a list that contains the user's preferences ["Hot", "Light", "Tangy"] about the product he wants to consume you should infer from this data for deciding the user preference upon which you should top 3 products to recommend there might be cases when intent list only contains one or two elements instead of three, in that case, you should consider only those preferences which are in the list and assuming other to be null. contextual data is also a list containing [time, date, location, weather] this is also an important part of our system. lets suppose according to the user's emotion and preference you have 5 items to recommend then use contextual data to re-arrange between them as first, second and third, the way you should do this is time, date, and weather. so basically see what time of the day it is and what type of food user should have at that particular time, from date you should find out if there is any festival in India on those particular day(here you can also see to 3 before and 3 after days of weeks as well if there is upcoming or there was any festival nearby this date), if yes then use this data as well for rearranging the recommendation, and from the weather use it to recommend the products that will suit at that particular weather(e.g. on sunny days recommend cooling products on the top). So basically you should use contextual data for final rearranging the products in between them so that we can come up with best top 3 recommendations. And keep in mind you should use contextual data only for rearranging between the products not deciding, products decision should be made only from VAD score and Intent list. Contextual data is just like oregano over pizza which enhance the taste of pizza but doesn't alter what it contain.

Product Dictionary: {
    "Malabar Parata": {
        "variants": [],
        "description": "A flaky, layered Indian bread usually served hot with pickles or curries. Malabar Parata is a beloved staple in South Indian coastal cuisine."
    }, 
    "Pickles": {
        "variants": [],
        "description": "Pickles are vegetables or fruits preserved in a solution of vinegar, brine, or other acidic mediums, often enhanced with spices. A delicious and satisfying food item with tangy, salty, and sometimes spicy flavor, suitable for snacks or meals, paired with Malabar Parata only."
    },
    "Chaach": {
        "variants": [
            "Taaza Jeera Chaach",
            "Taaza Pudina Masala Chaach"
        ],
        "description": "A traditional, spiced yogurt-based drink(made with yogurt, spices, and herbs also known as buttermilk) served chilled for digestive and cooling benefits. Known for its tangy cumin or mint undertones."
    },
    "Yogurt": {
        "variants": [
            "Mango Yogurt",
            "Blueberry Yogurt",
            "Strawberry Yogurt",
            "Mango Badam Drinking yogurt [Probiotic]",
            "Vanilla Badam Drinking Yogurt [Probiotic]",
            "Strawberry Badam Drinking Yogurt [Probiotic]"
        ],
        "description": "A creamy, refreshing, and tangy dairy snack or drink, rich in probiotics and great for any time of the day. Flavors range from sweet mango to tangy berries or nutty probiotic drinks. yogurt is a versatile and nutritious food."
    },
    "Chocolate": {
        "variants": [
            "Dark Chocolate 55% - Nuts Infused",
            "Dark Chocolate - 55%",
            "Milk Chocolate - Nutty & Creamy"
        ],
        "description": "A sweet indulgence made from cocoa bean, ideal for satisfying dessert cravings and lifting mood. Available in dark or milk forms, often blended with nuts."
    },
    "Butter": {
        "variants": [
            "Unsalted Butter",
            "Salted Butter"
        ],
        "description": "A delicious and satisfying food item suitable for snacks or meals. Paired only with Malabar Parata or any bread. Choose between rich salted or classic unsalted profiles."
    },
    "Bread": {
        "variants": [
            "Multigrain Bread",
            "White Bread",
            "Brown Bread"
        ],
        "description": "A baked staple available in different types, commonly used for toast or sandwiches, Paired with Butter, Jam, or Peanut butter. Multigrain offers fiber, white is soft, and brown is wholesome."
    },
    "Butter Atta Biscuit": {
        "variants": [],
        "description": "Crunchy or chewy baked snacks often paired with tea or coffee or with Milk only."
    },
    "Milk": {
        "variants": [
            "Chocolate Milk",
            "Kesar Milk"
        ],
        "description": "A nutrient-rich drink consumed plain or flavored, often paired with bread, cereals, or Muesli and Flakes. Flavored options like Kesar and Chocolate enhance indulgence."
    },
    "Muesli and Flakes": {
        "variants": [
            "Corn Flakes",
            "Muesli",
            "Chocolate Flakes",
            "Chocolate Muesli"
        ],
        "description": "A dry breakfast cereal typically consumed with milk, providing fiber and light energy to start your day. and chocolate ones are also ideal for satisfying dessert cravings or lifting mood. Chocolate versions add a dessert-like twist."
    },
    "Malai Peda": {
        "variants": [],
        "description": "Rich Indian sweets made from milk solids or lentils, usually enjoyed during festivals or as a 0dessert or for lifting mood. Made from reduced milk, offering a soft, melt-in-mouth texture."
    },
    "Lassi": {
        "variants": [
            "Mango Lassi",
            "Punjabi Lassi"
        ],
        "description": "A traditional, spiced yogurt-based drink served chilled for digestive and cooling benefits. Comes in fruity or traditional creamy styles."
    },
    "Gulab Jamun": {
        "variants": [],
        "description": "A fruity, sweet spread commonly enjoyed on bread or toast for breakfast, as a dessert or for lifting mood. Deep-fried dough balls soaked in cardamom-scented syrup."
    },
    "Kaju Katli": {
        "variants": [],
        "description": "Rich Indian sweets made from milk solids or lentils, usually enjoyed during festivals, as desserts, or for lifting mood. Smooth cashew fudge with a hint of cardamom."
    },
    "Honey": {
        "variants": [
            "Farm Honey",
            "Wild Forest Honey"
        ],
        "description": "A natural sweetener used in cooking or as a healthy sugar substitute in drinks and spreads. Farm honey is mild and sweet; Wild forest honey has deeper, earthy notes."
    },
    "Sweet Corn": {
        "variants": [],
        "description": "A delicious and satisfying food item suitable for snacks or meals. Rich in fiber, vitamins, and antioxidants, sweet corn is both a tasty and nutritious addition to meals."
    },
    "Sprouts": {
        "variants": [],
        "description": "Fresh and light ingredients typically used in salads or eaten raw for health benefits. Low-calorie, hydrating, and nutrient-packed."
    },
    "Cookies": {
        "variants": [
            "Kaju Pista Cookies",
            "Choco-Chip Cookies"
        ],
        "description": "Crunchy or chewy baked snacks often paired with tea, coffee, or milk. Kaju Pista is nutty and traditional, and Choco-chip is rich and indulgent."
    },
    "Jam": {
        "variants": [
            "Mixed Fruit Jam",
            "Strawberry Jam"
        ],
        "description": "A fruity, sweet spread commonly enjoyed on bread or toast for breakfast. Paired with bread only. Mixed fruit is vibrant and tangy; strawberry is a sweet classic."
    },
    "Coffee": {
        "variants": [
            "Instant 100% Coffee",
            "Instant Chicory Blend Coffee"
        ],
        "description": "A hot or cold brewed beverage known for its bold flavor and stimulating properties. Chicory blend adds depth, while 100% coffee is pure and bold."
    },
    "Makhana": {
        "variants": [
            "Roasted Makhana - Cheese",
            "Roasted Makhana - Ghee Turmeric",
            "Roasted Makhana - Himalayan Pink Salt",
            "Roasted Makhana - Salt & Pepper",
            "Roasted Makhana - Chatpata Pudina"
        ],
        "description": "Lightly roasted lotus seeds offering a crunchy and low-calorie snacking option, ideal for munching anytime. Flavored with herbs, cheese, or spice for variety."
    },
    "Chips": {
        "variants": [
            "Cream & Onion - Potato Chips",
            "Salted - Potato Chips",
            "Magic Masala - Potato Chips",
            "Hot 'N' Sweet Chilli - Potato Chips",
            "Tangy Tomato Chips",
            "Flaming Hot - Spicy(Potato Chips)",
            "Cheesy Chips",
            "Banana Chips"
        ],
        "description": "Crunchy, salty snacks flavored in various styles, ideal for munching anytime. Can be bundled with them themselves, Can be Suggested as a combo of any number of them, or in combo with No Maida Puffs, Krunchy Puffs Magic Masala, Krunchy Stix. Ranges from cheesy to spicy, with options like Tomato, Chilli, and Banana."
    },
    "No Maida Puffs": {
        "variants": [
            "Disney Cream & Onion Puffs",
            "Disney Noodle Masala Puffs"
        ],
        "description": "Crunchy, salty snacks flavored in various styles, ideal for munching anytime. Can be bundled with them themselves, Can be Suggested as a combo of any number of them, or in combo with Chips, Krunchy Puffs Magic Masala, Krunchy Stix. Comes in masala, mint, and noodle-inspired flavors."
    },
    "Krunchy Puffs Magic Masala": {
        "variants": [],
        "description": "Crunchy, salty snacks flavored in various styles, ideal for munching anytime. Can be bundled with them themselves, Can be Suggested as a combo of any number of them, or in combo with No Maida Puffs, Chips, Krunchy Stix."
    },
    "Krunchy Stix": {
        "variants": [
            "Noodle Masala Krunchy Stix",
            "Mint Chutney Krunchy Stix",
            "Magic Masala Krunchy Stix"
        ],
        "description": "Crunchy, salty snacks flavored in various styles, ideal for munching anytime. Can be bundled with them themselves, Can be Suggested as a combo of any number of them, or in combo with No Maida Puffs, Krunchy Puffs Magic Masala, Chips Comes in masala, mint, and noodle-inspired flavors."
    },
    "Chikki": {
        "variants": [
            "Peanut Chikki",
            "Peanut Chikki Gud Patti"
        ],
        "description": "A delicious and satisfying food item suitable for snacks, meals, or as dessert. Crunchy, sweet, and jaggery-based with roasted peanuts."
    },
    "Aloo Bhujia": {
        "variants": [],
        "description": "Savory, crunchy Indian snacks often enjoyed with tea, as snack or as appetizers. ideal for munching anytime. Spiced and deep-fried for maximum flavor impact."
    },
    "Dry Bhel": {
        "variants": [],
        "description": "A popular Indian snack made with puffed rice, sev (crunchy chickpea noodles), roasted peanuts, and a mix of tangy and spicy chutneys or masalas. Unlike its wetter counterpart, it’s served without tamarind sauce, making it crisp, light, and easy to enjoy on the go. Garnished with chopped onions, tomatoes, coriander, and a squeeze of lime. A delicious and satisfying food item suitable for snacks or meals. ideal for munching anytime."
    },
    "Spiced Coated Peanuts": {
        "variants": [],
        "description": "Spiced Coated Peanuts are a bold and crunchy snack made by roasting premium peanuts and coating them with a flavorful blend of spices. With a crispy outer layer and a savory, slightly spicy kick, perfect for snacking on the go. Irresistibly addictive, these peanuts offer the ideal balance of heat, crunch, and nutty richness. Ideal for munching anytime."
    },
    "Moong Dal Namkeen": {
        "variants": [],
        "description": "Savory, crispy, crunchy Indian snacks made from split green gram (moong dal) that is deep-fried to golden perfection and lightly seasoned with salt and spices. Known for its light texture and addictive crunch. Often enjoyed with tea, as snack, or as appetizer. and ideal for munching anytime. Spiced and deep-fried for maximum flavor impact."
    },
    "Mathri": {
        "variants": [
            "Ajwain Mathri",
            "Methi Mathri"
        ],
        "description": "Savory, crunchy Indian snack known for its crisp texture and rich flavor seasoned with spices like ajwain (carom seeds), black pepper, and sometimes kasuri methi. Often enjoyed with tea, as snack, or as appetizers, and is a staple during festivals. and ideal for munching anytime. Spiced and deep-fried for maximum flavor impact."
    },
    "Farsan": {
        "variants": [
            "Mix Namkeen"
        ],
        "description": "Savory, crunchy, and flavorful Indian snack blend, combining an assortment of savory fried lentils, chickpeas, nuts, and spices. Perfectly spiced and balanced, it offers a delightful mix of textures and tastes often enjoyed with tea, as snack, or as appetizer. and ideal for munching anytime, or on festive occasions."
    },
    "Salted and Roasted": {
        "variants": [
            "Salted Roasted Chana",
            "Salted Roasted Almonds"
        ],
        "description": "A delicious and satisfying food item suitable for snacks or meals. ideal for munching anytime. Nutty, high-protein options with a savory finish. Salted Roasted Chana are Crunchy, protein-packed roasted chickpeas lightly salted for a deliciously healthy and satisfying snack. whereas Salted Roasted Almonds are Toasty roasted almonds sprinkled with just the right amount of salt, offering a perfect balance of savory flavor and natural nuttiness."
    },
    "Chana Dal Masala": {
        "variants": [],
        "description": "Savory, crunchy, flavourful Indian snacks often enjoyed with tea, as snack or as appetizers. and ideal for munching anytime. This comforting dish is rich in protein and perfect as a nutritious vegetarian meal."
    },
    "Spicy Boondi": {
        "variants": [],
        "description": "A delicious and satisfying food item suitable for snacks or meals. ideal for munching anytime. Tiny fried gram balls coated with spices, often added to raita."
    },
    "All in One Mixture": {
        "variants": [],
        "description": "Savory, crunchy, flavorful Indian snacks blend combining a variety of crunchy ingredients—like sev, peanuts, boondi, and spiced lentils—into one irresistible mix. Perfectly seasoned with traditional Indian spices, it's the ideal balance of taste, texture, and satisfaction in every bite. Great for tea-time, parties, or anytime cravings.}
}
Product dictionary contains products as a dictionary which further contains particular product variants available with their short description (but not limited to if you can think if additional discription you can include it at the time of making recommendations). You should recommend only one variant of a particular product (If you can't decide between variants which one to recommend you can choose any one at random from all of those variants listed). For the top three products on recommendation only recommend products which are not combo with other(e.g. Muesli should always be recommended with milk never recommended alone, and Malabar parata should always be recommended with pickles never alone so these products should be recommended as combos other than top three products recommended and you also have to recommend top three combos with top three products(If there is no combo that you can think of from user input(i.e. VAD and intent) you can recommend any three combos at random.), in case of chips and puffs you can recommend them in bundles as single product as well. And one more thing during recommendation in case of products with variants use the name of variant on recommendation not the product name(e.g. if you are considering recommending chips never show chips in the recommendation output only write the final variant name that you decided to recommend.

Below are the use cases(but not limited to If you can think of more you should consider including them as well before making recommendations) they explain different cases and their different behavior but I am not limiting you to these use cases only you can use your own analysis and consider these use cases as well in your analysis, If you find and of these thing contradictory to you analysis you are free to go with your own analysis and ignore these use cases:
{
    "VAD_Dimensions": {
        "Valence": {
            "description": "Pleasantness of emotion",
            "scale": "-1 (very unpleasant) to +1 (very pleasant)"
        },
        "Arousal": {
            "description": "Energy or intensity level of emotion",
            "scale": "-1 (very calm) to +1 (very energized)"
        },
        "Dominance": {
            "description": "Sense of control over the emotional state",
            "scale": "-1 (helpless) to +1 (in control)"
        }
    },
    "Food_Influences_By_Dimension": {
        "Valence": {
            "High": {
                "emotions": [
                    "Joyful",
                    "Content"
                ],
                "food_behavior": "Celebratory or indulgent foods, sweets, mindful or social eating"
            },
            "Low": {
                "emotions": [
                    "Sad",
                    "Anxious",
                    "Bored"
                ],
                "food_behavior": "Comfort eating: high sugar, fat, salt, crunchy or chewy textures"
            }
        },
        "Arousal": {
            "High": {
                "emotions": [
                    "Excited",
                    "Stressed",
                    "Angry"
                ],
                "food_behavior": "Quick snacks, crunchy textures, emotional or rapid eating"
            },
            "Low": {
                "emotions": [
                    "Calm",
                    "Bored",
                    "Depressed"
                ],
                "food_behavior": "Comfort food, grazing/snacking, boredom eating or mindful eating"
            }
        },
        "Dominance": {
            "High": {
                "emotions": [
                    "Confident",
                    "In control"
                ],
                "food_behavior": "Mindful, planned, healthy or goal-aligned choices"
            },
            "Low": {
                "emotions": [
                    "Helpless",
                    "Overwhelmed"
                ],
                "food_behavior": "Impulsive or out-of-control eating, often comfort food"
            }
        }
    },
    "Combined_VAD_Profiles": {
        "HighV_HighA_HighD": {
            "emotions": [
                "Excited",
                "Confident"
            ],
            "food_behavior": "Celebratory, rich food, enjoyed socially"
        },
        "HighV_LowA_HighD": {
            "emotions": [
                "Content",
                "Peaceful"
            ],
            "food_behavior": "Mindful, healthy eating"
        },
        "LowV_HighA_LowD": {
            "emotions": [
                "Anxious",
                "Stressed",
                "Helpless"
            ],
            "food_behavior": "Emotional or binge eating, comfort food"
        },
        "LowV_LowA_LowD": {
            "emotions": [
                "Depressed",
                "Bored",
                "Powerless"
            ],
            "food_behavior": "Mindless comfort eating, sweet or heavy foods"
        },
        "LowV_HighA_HighD": {
            "emotions": [
                "Frustrated",
                "Angry"
            ],
            "food_behavior": "Crunchy food for tension release, variable appetite"
        },
        "LowV_LowA_HighD": {
            "emotions": [
                "Resigned",
                "Low Mood"
            ],
            "food_behavior": "Unhealthy choices with controlled quantity"
        }
    },
    "Food_Type_Mapping": {
        "Sweet": {
            "typical_VAD": [
                "LowV",
                "LowA or HighA"
            ],
            "reason": "Boost mood, reward system, serotonin uplift"
        },
        "Salty_Tangy": {
            "typical_VAD": [
                "HighA",
                "LowV"
            ],
            "reason": "Stress relief, sensory distraction, crunch for tension"
        },
        "Crunchy": {
            "typical_VAD": [
                "HighA",
                "LowD"
            ],
            "reason": "Physical release of tension or dominance compensation"
        },
        "Light": {
            "typical_VAD": [
                "HighV",
                "LowA",
                "HighD"
            ],
            "reason": "Health-aligned, mindful, soothing in high stress"
        },
        "Heavy": {
            "typical_VAD": [
                "LowV",
                "LowD",
                "Any A"
            ],
            "reason": "Comfort, fullness, grounding against helplessness or sadness"
        }
    }
}
In case of some bottlenecks you should proceed in the way below:
1. In case of user requesting "light" product while having a VAD profile that suggests a "Heavy" product. You should give preference to user preference of light product instead of VAD profile suggesting Heavy product.
2. In case of user has multiple intents (e.g., ["Hot", "Light", "Healthy"]), the system might struggle to prioritize or combine these intents accurately. here you can choose which intent to prioritize and which one to drop, if you are struggling to recommend product based on these intent. BUT don't make it a habit of dropping or ignoring intent consider all of the intents wherever possible.
3. The system categorizes products into types (e.g., Sweet, Salty, Crunchy). However, some products might belong to multiple categories or have ambiguous categorization, leading to inconsistent recommendations: In this case where the product belongs to multiple categories you can use them recommending into multiple profiles as well instead of only listed types.
4. In case of multiple variants where system struggle to choose between them, it can decide randomly between the variants.
5. In case of generating combos if you are unable to find combos matching to user's VAD and intent, you can decide other combos randomly and recommend them.
6. Use contextual data only to rearrange the recommended products, If you can't use contextual data it is not a must but use contextual data wherever possible.

In case of recommending products and combos which contain variants don't recommend as "Makhana - Roasted Makhana" Instead recommend only "Roasted Makhana". And in case of Combos there are very randomness (i.e. products are combined without any match between them(e.g. Mango Yogurt & Muesli and Flakes - Muesli where Mango Yogurt has no combo with Muesli.)) In the product dictionary there is explicitly mentioned which product can be combined with other products. So I want you to use that information to recommend combos not any random combination to make.


In the output what you must include is(and the output should be in JSON):
1. Emotion of the user from the VAD score limiting it to only two words (i.e. those two words should be self-explanatory about the user's emotion). If there is a way to write it in only one word then there is no need to write two words for emotion.
2. Top three products recommended as a list.
3. Top three combos recommended as a list.
"""
                },
                {
                    "role": "user",
                    "content": str(input_data)
                }
            ],
            temperature=2,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"},
            stop=None,
        )
        
        # Extract the response content
        response_content = completion.choices[0].message.content
        
        # Parse the JSON response
        recommendations = json.loads(response_content)
        
        # Normalize keys to ensure consistent capitalization
        normalized_recommendations = {}
        
        # Convert keys to expected format (first letter capitalized)
        for key, value in recommendations.items():
            if key.lower() == 'emotion':
                normalized_key = 'Emotion'
            elif key.lower() == 'top products':
                normalized_key = 'Top Products'
            elif key.lower() == 'top combos':
                normalized_key = 'Top Combos'
            else:
                normalized_key = key
            
            normalized_recommendations[normalized_key] = value
        
        # Ensure emotion data is properly formatted
        if "Emotion" in normalized_recommendations and isinstance(normalized_recommendations["Emotion"], list):
            # Make sure each emotion word is a single string without spaces between characters
            normalized_recommendations["Emotion"] = [word.strip() for word in normalized_recommendations["Emotion"]]
        elif "Emotion" in normalized_recommendations and isinstance(normalized_recommendations["Emotion"], str):
            # If emotion is a string, convert it to a list of words
            normalized_recommendations["Emotion"] = [word.strip() for word in normalized_recommendations["Emotion"].split()]
        
        # Log the recommendations
        logger.info(f"Food recommendations received: {normalized_recommendations}")
        
        return normalized_recommendations
        
    except Exception as e:
        logger.error(f"Error in get_food_recommendations: {str(e)}")
        # Return a default response in case of error
        return {
            "Emotion": ["Unknown", "Error"],
            "Top Products": ["Error retrieving products", "Please try again", "Check connection"],
            "Top Combos": ["Error retrieving combos", "Please try again", "Check connection"]
        }
