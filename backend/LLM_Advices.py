import json
import random

def generate_energy_advice(facility_data: dict) -> dict:
    """
    Generates personalized utility-saving messages for 4 horizons independently.
    """
    try:
        resource = facility_data.get("resource", "Electricity").capitalize()
        environment = facility_data.get("environment_type", "Company").capitalize()
        company_type = facility_data.get("facility_subtype", "Office").capitalize()
        company_size = facility_data.get("facility_size", "Medium").lower()
        
        predictions = facility_data.get("predictions", {})
        waste_analysis = facility_data.get("waste_analysis", {})
        
        current_category = facility_data.get("historical_analytics", {}).get("category", 1)
        if isinstance(current_category, str):
            import re
            match = re.search(r'\d+', current_category)
            current_category = int(match.group()) if match else 1
            
        unit = "kWh" if resource == "Electricity" else "m³"

        # Equipment lists based on provided instructions
        equipment_house_elec = ["refrigerator", "deep freezer", "washing machine", "dishwasher", "electric kettle", "water heater", "boiler", "air conditioner", "electric oven", "microwave", "television", "water pump", "lighting"]
        equipment_house_water = ["taps", "shower", "toilet", "washing machine", "dishwasher", "water heater", "water tank", "water pump", "garden irrigation"]
        equipment_house_gas = ["gas cooker", "gas oven", "gas water heater", "gas boiler", "central heating equipment"]

        equipment_bakery_elec = ["bakery oven", "dough mixer", "dough kneader", "refrigerator", "deep freezer", "proofing machine", "electric kettle", "boiler", "water heater", "ventilation system", "air conditioner", "water pump"]
        equipment_bakery_water = ["cleaning equipment", "taps", "sinks", "water tanks", "water pumps", "dough preparation equipment", "dishwashing equipment"]
        equipment_bakery_gas = ["gas bakery oven", "gas burner", "gas boiler", "gas water heater", "heating equipment"]
        
        equipment_office_elec = ["desktop computers", "laptops", "printers", "photocopiers", "air conditioners", "water cooler", "electric kettle", "refrigerator", "microwave", "server/network equipment", "lighting"]
        equipment_office_water = ["toilets", "taps", "cleaning systems"]
        equipment_office_gas = ["gas boiler", "heating equipment"]
        
        equipment_hotel_elec = ["air conditioners", "refrigerators", "deep freezers", "boilers", "water heaters", "washing machines", "dryers", "dishwashers", "kitchen ovens", "water pumps", "elevators", "lighting"]
        equipment_hotel_water = ["showers", "toilets", "taps", "washing machines", "dishwashers", "water heaters", "water tanks", "swimming-pool systems", "irrigation systems"]
        equipment_hotel_gas = ["gas boilers", "gas water heaters", "kitchen ovens", "gas cookers", "commercial burners", "heating systems"]
        
        equipment_rest_elec = ["commercial oven", "electric fryer", "refrigerator", "deep freezer", "dishwasher", "food warmer", "electric kettle", "boiler", "water heater", "exhaust hood", "air conditioner", "water pump", "lighting"]
        equipment_rest_water = ["kitchen sinks", "dishwashers", "taps", "food-cleaning stations", "water heaters", "water tanks", "water pumps", "cleaning systems"]
        equipment_rest_gas = ["gas oven", "gas cooker", "commercial burners", "gas fryer", "gas boiler", "gas water heater"]
        
        equipment_school_elec = ["air conditioners", "computers", "projectors", "printers", "refrigerators", "deep freezers", "water coolers", "electric kettles", "boilers", "water heaters", "water pumps", "lighting"]
        equipment_school_water = ["toilets", "taps", "drinking-water systems", "water tanks", "water pumps", "cleaning equipment", "irrigation systems"]
        equipment_school_gas = ["gas boiler", "gas water heater", "kitchen cooker", "kitchen oven", "heating equipment"]
        
        equipment_supermarket_elec = ["display refrigerators", "commercial freezers", "deep freezers", "refrigeration systems", "air conditioners", "bakery ovens", "water pumps", "electric doors", "lighting", "POS systems", "computers"]
        equipment_supermarket_water = ["cleaning systems", "taps", "toilets", "water tanks", "water pumps", "food-preparation equipment", "irrigation systems"]
        equipment_supermarket_gas = ["gas boiler", "gas water heater", "bakery oven", "food-preparation equipment"]
        
        def get_equipment_list(env, c_type, res):
            if env == "House":
                if res == "Electricity": return equipment_house_elec
                elif res == "Water": return equipment_house_water
                else: return equipment_house_gas
            else:
                if c_type == "Bakery":
                    if res == "Electricity": return equipment_bakery_elec
                    elif res == "Water": return equipment_bakery_water
                    else: return equipment_bakery_gas
                elif c_type == "Hotel":
                    if res == "Electricity": return equipment_hotel_elec
                    elif res == "Water": return equipment_hotel_water
                    else: return equipment_hotel_gas
                elif c_type == "Restaurant":
                    if res == "Electricity": return equipment_rest_elec
                    elif res == "Water": return equipment_rest_water
                    else: return equipment_rest_gas
                elif c_type == "School":
                    if res == "Electricity": return equipment_school_elec
                    elif res == "Water": return equipment_school_water
                    else: return equipment_school_gas
                elif c_type == "Supermarket" or c_type == "SuperMarket":
                    if res == "Electricity": return equipment_supermarket_elec
                    elif res == "Water": return equipment_supermarket_water
                    else: return equipment_supermarket_gas
                else:
                    if res == "Electricity": return equipment_office_elec
                    elif res == "Water": return equipment_office_water
                    else: return equipment_office_gas
                    
        eq_list = get_equipment_list(environment, company_type, resource)
        
        def get_equipment_str(seed_val, count=3):
            random.seed(seed_val)
            sample = random.sample(eq_list, min(count, len(eq_list)))
            if len(sample) > 1:
                return ", ".join(sample[:-1]) + ", and " + sample[-1]
            elif len(sample) == 1:
                return sample[0]
            else:
                return "general equipment"

        def get_previous_category_max(target_period):
            if environment == "House":
                if resource == "Electricity":
                    thresholds = [50, 100, 200, 350, 650, 1000]
                elif resource == "Water":
                    thresholds = [10, 20, 30]
                else:
                    thresholds = [30, 60]
            else:
                if resource == "Electricity":
                    thresholds = [100, 250, 600, 1000]
                elif resource == "Water":
                    thresholds = [50, 100]
                else:
                    thresholds = [100, 250]

            mult = 1.0
            if target_period == 'next_day': mult = 1/30.0
            elif target_period == 'next_week': mult = 7/30.0
            elif target_period == 'next_quarter': mult = 3.0

            if current_category <= 1 or current_category > len(thresholds) + 1:
                return -1
            return thresholds[current_category - 2] * mult

        output = {}
        horizons = ["next_day", "next_week", "next_month", "next_quarter"]
        
        for horizon in horizons:
            # Map horizon to period string for waste dicts (day, week, month, quarter)
            period = horizon.split('_')[1]
            
            consumption = predictions.get(horizon, 0) or 0
            waste_amt = waste_analysis.get(f"waste_{period}", 0) or 0
            waste_cost = waste_analysis.get(f"waste_cost_{period}", 0) or 0
            
            if consumption > 0:
                waste_pct = (waste_amt / consumption) * 100
            else:
                waste_pct = 0
            
            eq_str_1 = get_equipment_str(horizon + "1", 3)
            eq_str_2 = get_equipment_str(horizon + "2", 3)
            eq_str_3 = get_equipment_str(horizon + "3", 3)
            
            # 1. Waste Reduction Message
            if waste_pct == 0:
                w_msg = f"No measurable waste is predicted for the {horizon.replace('_', ' ')}. Maintain your current efficiency and keep monitoring your usage to ensure no unnecessary consumption."
            elif 0 < waste_pct <= 20:
                w_msg = f"Your predicted waste is low ({waste_pct:.1f}%, {waste_amt:.1f} {unit}). To prevent this waste, which could cost {waste_cost:.2f}, turn off unused equipment such as the {eq_str_1}. Reduce standby consumption and avoid unnecessary operation."
            elif 20 < waste_pct <= 40:
                w_msg = f"Your predicted waste is moderate ({waste_pct:.1f}%, {waste_amt:.1f} {unit}), with an estimated cost of {waste_cost:.2f}. Reduce unnecessary operating hours and optimize schedules for equipment like the {eq_str_1}. Turn off equipment when not required."
            elif 40 < waste_pct <= 60:
                w_msg = f"Your predicted waste is high ({waste_pct:.1f}%, {waste_amt:.1f} {unit}), representing a potential cost of {waste_cost:.2f}. Focus on high-consumption equipment such as the {eq_str_1}. Review operating schedules, perform preventive maintenance, and check for unnecessary standby consumption."
            elif 60 < waste_pct <= 80:
                w_msg = f"Your predicted waste is very high ({waste_pct:.1f}%, {waste_amt:.1f} {unit}), which could cost you {waste_cost:.2f}. Prioritize checking the {eq_str_1}. Identify the main sources of waste and take steps to reduce unnecessary consumption immediately."
            elif 80 < waste_pct <= 100:
                w_msg = f"Your predicted waste is extremely high ({waste_pct:.1f}%, {waste_amt:.1f} {unit}). The predicted waste represents an estimated cost of {waste_cost:.2f}. Take immediate action: investigate your equipment such as the {eq_str_1}, implement operational improvements, and monitor your usage closely."
            else:
                w_msg = f"Your predicted waste is exceptionally high ({waste_pct:.1f}%, {waste_amt:.1f} {unit}), representing a severe potential cost of {waste_cost:.2f}. Investigate equipment operation and verify meter readings. Look for abnormal behavior in the {eq_str_1}. If this continues, we recommend inspection by a qualified professional."
            
            # 2. Consumption Reduction Message
            if current_category == 1:
                c_msg = f"You are currently in Category 1 for this horizon, with a predicted consumption of {consumption:.1f} {unit}. You are already in the most efficient category. Keep up the good work and maintain your efficient habits to avoid higher costs."
            else:
                prev_max = get_previous_category_max(horizon)
                if prev_max > 0 and consumption > prev_max:
                    req_reduction = consumption - prev_max
                    c_msg = f"Your predicted consumption is {consumption:.1f} {unit} (Category {current_category}). You need to reduce your consumption by approximately {req_reduction:.1f} {unit} to reach Category {current_category - 1}. Consider optimizing the usage of the {eq_str_2}. Reducing this predicted waste could help lower your cost by approximately {waste_cost:.2f}."
                else:
                    c_msg = f"Your predicted consumption is {consumption:.1f} {unit} (Category {current_category}). To reduce consumption toward Category {current_category - 1}, consider optimizing the usage of the {eq_str_2}. Reducing this predicted waste could help lower your cost by approximately {waste_cost:.2f}."

            # 3. Equipment / Malfunction Check
            e_msg = f"Based on a predicted waste of {waste_amt:.1f} {unit}, this could indicate inefficient operation. Consider checking equipment such as the {eq_str_3}. It may be useful to inspect them for faults or incorrect settings. If the abnormal consumption continues, have the equipment inspected by a qualified technician."
            
            # 4. Smart Saving Advice (Horizon Specific)
            if horizon == "next_day":
                s_msg = "Focus on immediate actions: Turn off unnecessary equipment before leaving, adjust equipment settings for tomorrow, and prevent unnecessary standby consumption."
            elif horizon == "next_week":
                s_msg = "Review weekly operating behavior: Identify equipment that remains active unnecessarily, create a weekly shutdown routine, and schedule preventive checks."
            elif horizon == "next_month":
                s_msg = "Focus on recurring consumption patterns: Set a monthly consumption target, optimize operating schedules, and perform preventive maintenance on high-consumption equipment."
            elif horizon == "next_quarter":
                s_msg = "Focus on long-term optimization: Perform a utility consumption review, establish quarterly consumption targets, and consider strategic operational improvements."
                
            if environment == "Company":
                if company_size == "small":
                    s_msg += " As a small business, prioritize simple operational changes, turning off unused equipment, and low-cost improvements."
                elif company_size == "medium":
                    s_msg += " For a medium-sized operation, focus on equipment scheduling, employee awareness, and monitoring high-consumption equipment."
                elif company_size == "large":
                    s_msg += " As a large facility, implement systematic utility management, equipment-level monitoring, automated controls, and department-level consumption tracking."
            else:
                s_msg += " For your household, focus on daily habits, household appliances, and lighting."

            title_horizon = horizon.replace('_', ' ').title()
            if title_horizon == "Next Day":
                title_horizon_w = "Tomorrow's"
                title_horizon_c = "Tomorrow's"
            elif title_horizon == "Next Week":
                title_horizon_w = "This Week's"
                title_horizon_c = "This Week's"
            elif title_horizon == "Next Month":
                title_horizon_w = "This Month's"
                title_horizon_c = "This Month's"
            else:
                title_horizon_w = "This Quarter's"
                title_horizon_c = "This Quarter's"

            if horizon == "next_day":
                s_title = "Daily Saving Tip"
            elif horizon == "next_week":
                s_title = "Weekly Saving Tip"
            elif horizon == "next_month":
                s_title = "Monthly Saving Tip"
            else:
                s_title = "Quarterly Saving Tip"

            c_type_value = "category_maintenance" if current_category == 1 else "category_reduction"
            c_title = f"Maintain {title_horizon_c} Efficiency" if current_category == 1 else f"Reduce {title_horizon_c} Consumption"

            messages = [
                {
                    "type": "waste_reduction",
                    "title": f"Reduce {title_horizon_w} Waste",
                    "message": w_msg.replace(". ", ".\n\n")
                }
            ]
            
            if not (resource == "Gas" and environment == "Company"):
                messages.append({
                    "type": c_type_value,
                    "title": c_title,
                    "message": c_msg.replace(". ", ".\n\n")
                })

            messages.extend([
                {
                    "type": "equipment_check",
                    "title": "Check Your Equipment",
                    "message": e_msg.replace(". ", ".\n\n")
                },
                {
                    "type": "smart_advice",
                    "title": s_title,
                    "message": s_msg.replace(". ", ".\n\n")
                }
            ])
            
            output[horizon] = {"messages": messages}
            
        return output
    except Exception as e:
        print(f"Error in generating advice: {e}")
        return {}
