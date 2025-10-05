"""
FIA Regulations Evaluation Dataset

This module creates comprehensive evaluation datasets for testing
the FIA regulations agent across different intent categories and tools.
"""

import pandas as pd
from typing import List, Dict, Any
from datasets import Dataset


class FIAEvaluationDataset:
    """
    Creates evaluation datasets for FIA regulations agent testing.
    """
    
    def __init__(self):
        self.datasets = {}
    
    def create_regulation_search_dataset(self) -> Dataset:
        """Create dataset for regulation search tool evaluation."""
        
        data = [
            {
                "question": "What are the safety requirements for Formula 1 cars?",
                "ground_truth": "Formula 1 cars must have specific safety structures including survival cells, roll structures, cockpit padding, safety harnesses, fire extinguishers, and rear view mirrors. These are detailed in Article 12 (Car Construction and Survival Cell) and Article 14 (Safety Equipment) of the FIA Technical Regulations.",
                "contexts": [
                    "Article 12: Car Construction and Survival Cell - The survival cell is the continuous closed structure containing the fuel tank, the cockpit and the parts of the Energy Store. The lower plate of the Energy Store assembly is considered to be part of the Survival cell. Cockpit padding includes non-structural parts placed within the cockpit for the sole purpose of improving driver comfort and safety.",
                    "Article 14: Safety Equipment - All cars must be fitted with a fire extinguishing system which will discharge into the cockpit and into the engine compartment. The fire extinguishing system must be approved according to FIA standards. Cars must also have rear view mirrors, safety harnesses, and rear lights.",
                    "Safety requirements include impact testing following FIA Test Procedure 01/00 and specific load tests for survival cell frontal impact, roll structure testing, and side impact structures."
                ]
            },
            {
                "question": "What are the engine power limits in Formula 1?",
                "ground_truth": "Formula 1 engines are limited by rev limits rather than explicit power output. The engine high rev limits must remain within a band of 750rpm as specified in Article C5.14 of the 2026 Technical Regulations. Additional operational constraints are outlined in Articles C5.15 and C5.16.",
                "contexts": [
                    "Article C5.14: Engine high rev limits must remain within a band of 750rpm...",
                    "Article C5.15: Starting the engine procedures and constraints...",
                    "Article C5.16: Stall prevention systems and operational limits..."
                ]
            },
            {
                "question": "Find Article 5 about power units",
                "ground_truth": "Article 5 in the FIA Technical Regulations covers Power Units and includes sections on definitions (5.1), engine specifications (5.2), other means of propulsion and energy recovery (5.3), mass and centre of gravity (5.5), and energy recovery systems (5.13).",
                "contexts": [
                    "Article 5.1: Definitions - Power train: The power unit and associated torque transmission systems, up to but not including the drive shafts. Power unit (PU): The internal combustion engine and turbocharger, complete with its ancillaries, any energy recovery system and all actuation systems and PU-Control electronics necessary to make them function at all times.",
                    "Article 5.2: Engine specification - Only 4-stroke engines with reciprocating pistons are permitted. Engine cubic capacity must be 1600cc (+0/-10cc). Fuel mass flow must not exceed 100kg/h. Pressure charging may only be effected by the use of a sole single stage compressor with a single stage turbine.",
                    "Article 5.13: Energy Recovery System (ERS) - The system will be considered shut down when no high voltage can be present on any external or accessible part of the ERS, or across any capacitor belonging to the MGU control units."
                ]
            }
        ]
        
        return Dataset.from_list(data)
    
    def create_regulation_comparison_dataset(self) -> Dataset:
        """Create dataset for regulation comparison tool evaluation."""
        
        data = [
            {
                "question": "Compare Article 5 between 2024 and 2025",
                "ground_truth": "Article 5 in 2024 Technical Regulations covers Power Units with sections on definitions, engine specifications, energy recovery systems, mass requirements, and ERS specifications. The 2025 version may have updates or modifications reflecting technological advancements or regulatory changes.",
                "contexts": [
                    "2024 Article 5: Power Unit regulations including definitions, engine specs, and ERS requirements...",
                    "2025 Article 5: Updated power unit regulations with potential modifications...",
                    "Comparison shows evolution of power unit regulations between years..."
                ]
            },
            {
                "question": "What changed in safety regulations from 2024 to 2025?",
                "ground_truth": "Safety regulations in Formula 1 typically evolve to incorporate new safety technologies and lessons learned. Changes may include updates to survival cell requirements, enhanced safety equipment standards, and improved crash testing procedures.",
                "contexts": [
                    "2024 Safety Regulations: Article 12 and 14 requirements...",
                    "2025 Safety Regulations: Updated safety standards and requirements...",
                    "Evolution of safety regulations to improve driver protection..."
                ]
            }
        ]
        
        return Dataset.from_list(data)
    
    def create_penalty_lookup_dataset(self) -> Dataset:
        """Create dataset for penalty lookup tool evaluation."""
        
        data = [
            {
                "question": "What are the penalties for track limits violations?",
                "ground_truth": "Track limits violations in Formula 1 can result in various penalties including 5-second penalties (Article B1.10.4a), 10-second penalties (Article B1.10.4b), drive-through penalties (Article B1.10.4c), and stop-and-go penalties (Article B1.10.4d). Additional fines may apply for pit lane speeding.",
                "contexts": [
                    "Article 4.2: With the exception of a reprimand or fine, when a penalty is applied under the Code or Article 54.3 the stewards may impose penalty points on a driver's Super Licence. If a driver accrues twelve (12) penalty points his licence will be suspended for the following Competition.",
                    "Article 17.3: Appeals may not be made against decision concerning penalties imposed under Articles 54.3a), 54.3b), 54.3c), 54.3d), 54.3e), 54.3f) or 54.3g), including those imposed during the last three (3) laps or after the end of a sprint session or a race.",
                    "Penalty points will remain on a driver's Super Licence for a period of twelve (12) months after which they will be respectively removed on the twelve (12) month anniversary of their imposition."
                ]
            },
            {
                "question": "What penalties apply for fuel flow violations?",
                "ground_truth": "Fuel flow violations in Formula 1 can result in disqualification, time penalties, or grid position penalties depending on the severity. The specific penalties are outlined in the sporting regulations and can include exclusion from the race results.",
                "contexts": [
                    "Article 5.2.3: Fuel mass flow must not exceed 100kg/h. Below 10500rpm the fuel mass flow must not exceed Q (kg/h) = 0.009 N(rpm)+ 5.5. At partial load, the fuel mass flow must not exceed the limit curve defined in the regulations.",
                    "Article 17.3: Appeals may not be made against decision concerning penalties imposed under Articles 54.3a), 54.3b), 54.3c), 54.3d), 54.3e), 54.3f) or 54.3g), including those imposed during the last three (3) laps or after the end of a sprint session or a race.",
                    "Fuel flow violations can result in disqualification, time penalties, or grid position penalties depending on the severity. The specific penalties are outlined in the sporting regulations and can include exclusion from the race results."
                ]
            }
        ]
        
        return Dataset.from_list(data)
    
    def create_regulation_summary_dataset(self) -> Dataset:
        """Create dataset for regulation summary tool evaluation."""
        
        data = [
            {
                "question": "Summarize all safety requirements for Formula 1 cars",
                "ground_truth": "Formula 1 safety requirements encompass multiple areas: 1) Safety structures including survival cells and roll structures (Article 12), 2) Safety equipment like fire extinguishers, rear view mirrors, safety harnesses (Article 14), 3) Materials compliance with safety standards (Article 15), and 4) Fuel and engine oil safety requirements (Article 16). All components must undergo rigorous testing and homologation.",
                "contexts": [
                    "Article 12: Car Construction and Survival Cell - The survival cell is the continuous closed structure containing the fuel tank, the cockpit and the parts of the Energy Store. The lower plate of the Energy Store assembly is considered to be part of the Survival cell. Cockpit padding includes non-structural parts placed within the cockpit for the sole purpose of improving driver comfort and safety.",
                    "Article 14: Safety Equipment - All cars must be fitted with a fire extinguishing system which will discharge into the cockpit and into the engine compartment. The fire extinguishing system must be approved according to FIA standards. Cars must also have rear view mirrors, safety harnesses, and rear lights.",
                    "Article 15: Materials - General principles for permitted materials, specific prohibitions, and prescribed laminates for safety compliance.",
                    "Article 16: Fuel and Engine Oil - Basic principles for fuel definitions, fuel properties, composition of the fuel, fuel approval, and engine oil specifications for safety compliance."
                ]
            },
            {
                "question": "Provide a comprehensive overview of financial regulations",
                "ground_truth": "FIA Financial Regulations establish budget caps, spending limits, and financial monitoring for Formula 1 teams. These regulations ensure competitive balance and financial sustainability by limiting team expenditures on car development, operations, and personnel costs while maintaining transparency and compliance monitoring.",
                "contexts": [
                    "Budget cap regulations establish spending limits for Formula 1 teams to ensure competitive balance and financial sustainability. These limits cover car development, operations, and personnel costs.",
                    "Financial monitoring and compliance requirements include regular reporting, auditing procedures, and oversight mechanisms to ensure teams adhere to spending limits and maintain financial transparency.",
                    "Transparency and reporting obligations require teams to provide detailed financial information, undergo regular audits, and maintain compliance with established budget caps and spending regulations."
                ]
            }
        ]
        
        return Dataset.from_list(data)
    
    def create_general_rag_dataset(self) -> Dataset:
        """Create dataset for general RAG tool evaluation."""
        
        data = [
            {
                "question": "What are the technical specifications for Formula 1 engines?",
                "ground_truth": "Formula 1 engines must comply with specific technical regulations including displacement limits, fuel flow restrictions, energy recovery system specifications, and operational constraints. These are detailed in the FIA Technical Regulations with specific articles covering power unit requirements.",
                "contexts": [
                    "Article 5.2: Engine specification - Only 4-stroke engines with reciprocating pistons are permitted. Engine cubic capacity must be 1600cc (+0/-10cc). Fuel mass flow must not exceed 100kg/h. Pressure charging may only be effected by the use of a sole single stage compressor with a single stage turbine.",
                    "Article 5.2.3: Fuel mass flow must not exceed 100kg/h. Below 10500rpm the fuel mass flow must not exceed Q (kg/h) = 0.009 N(rpm)+ 5.5. At partial load, the fuel mass flow must not exceed the limit curve defined in the regulations.",
                    "Article 5.13: Energy Recovery System (ERS) - The system will be considered shut down when no high voltage can be present on any external or accessible part of the ERS, or across any capacitor belonging to the MGU control units."
                ]
            },
            {
                "question": "How do qualifying sessions work in Formula 1?",
                "ground_truth": "Formula 1 qualifying sessions determine the starting grid for the race. The format typically includes three sessions (Q1, Q2, Q3) with elimination rounds, where the slowest drivers are eliminated in each session. The fastest driver in Q3 starts on pole position.",
                "contexts": [
                    "Article 39: Qualifying and Sprint Qualifying Sessions - The format includes three sessions (Q1, Q2, Q3) with elimination rounds, where the slowest drivers are eliminated in each session based on their lap times.",
                    "Article 42: The Grid for the Sprint or the Race - Grid positions are determined by qualifying session results, with the fastest driver in Q3 starting on pole position. The grid formation follows specific procedures outlined in the sporting regulations.",
                    "Qualifying regulations include time limits for each session, elimination procedures, and specific rules for grid position determination based on lap times and session performance."
                ]
            }
        ]
        
        return Dataset.from_list(data)
    
    def create_out_of_scope_dataset(self) -> Dataset:
        """Create dataset for out-of-scope handler evaluation."""
        
        data = [
            {
                "question": "What is the weather today?",
                "ground_truth": "I cannot provide weather information as I am a specialized FIA Formula 1 regulations assistant. I can only help with questions about FIA regulations including technical, sporting, financial, and operational aspects.",
                "contexts": [
                    "I'm a specialized FIA Formula 1 regulations assistant...",
                    "I can only help with questions about FIA regulations...",
                    "Please ask me about FIA regulations instead..."
                ]
            },
            {
                "question": "How do I cook pasta?",
                "ground_truth": "I cannot provide cooking instructions as I am a specialized FIA Formula 1 regulations assistant. I can only help with questions about FIA regulations including technical, sporting, financial, and operational aspects.",
                "contexts": [
                    "I'm a specialized FIA Formula 1 regulations assistant...",
                    "I can only help with questions about FIA regulations...",
                    "Please ask me about FIA regulations instead..."
                ]
            }
        ]
        
        return Dataset.from_list(data)
    
    def create_comprehensive_dataset(self) -> Dict[str, Dataset]:
        """Create comprehensive evaluation dataset for all tools."""
        
        self.datasets = {
            "regulation_search": self.create_regulation_search_dataset(),
            "regulation_comparison": self.create_regulation_comparison_dataset(),
            "penalty_lookup": self.create_penalty_lookup_dataset(),
            "regulation_summary": self.create_regulation_summary_dataset(),
            "general_rag": self.create_general_rag_dataset(),
            "out_of_scope": self.create_out_of_scope_dataset()
        }
        
        return self.datasets
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about all datasets."""
        
        if not self.datasets:
            self.create_comprehensive_dataset()
        
        info = {}
        for name, dataset in self.datasets.items():
            info[name] = {
                "size": len(dataset),
                "columns": dataset.column_names,
                "sample_question": dataset[0]["question"] if len(dataset) > 0 else None
            }
        
        return info


def create_fia_evaluation_dataset() -> FIAEvaluationDataset:
    """Create and return FIA evaluation dataset."""
    return FIAEvaluationDataset()
