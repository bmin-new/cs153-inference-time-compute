import os
from mistralai import Mistral
import discord
from collections import defaultdict
from yelpapi import YelpAPI
import re
import logging
from datetime import datetime

MISTRAL_MODEL = "mistral-large-latest"
SYSTEM_PROMPT = "You are a helpful assistant that helps users find local businesses on Yelp and automate email outreach to get the best offer for a specific task or service."
MAX_HISTORY = 30

class MistralAgent:
    def __init__(self):
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        YELP_API_KEY = os.getenv("YELP_API_KEY")
        
        if not YELP_API_KEY:
            raise ValueError("YELP_API_KEY environment variable is not set")
            
        self.client = Mistral(api_key=MISTRAL_API_KEY)
        self.yelp_client = YelpAPI(api_key=YELP_API_KEY)
        self.channel_history = defaultdict(list)

    def _format_messages(self, messages):
        """Format Discord messages into the format expected by Mistral's API."""
        formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Process messages in chronological order
        for msg in messages:
            role = "assistant" if msg.author.bot else "user"
            formatted_messages.append({
                "role": role,
                "content": msg.content
            })
        
        return formatted_messages

    async def run(self, message: discord.Message):
        # Get message history from the channel
        channel = message.channel
        channel_id = channel.id
        
        # If we don't have history for this channel, fetch it
        if not self.channel_history[channel_id]:
            async for msg in channel.history(limit=MAX_HISTORY):
                self.channel_history[channel_id].append(msg)
            # Add system prompt to history
            self.channel_history[channel_id].insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        
        # Add the current message to history
        self.channel_history[channel_id].append(message)
        
        # Keep only the last MAX_HISTORY messages
        if len(self.channel_history[channel_id]) > MAX_HISTORY:
            self.channel_history[channel_id] = self.channel_history[channel_id][-MAX_HISTORY:]
        
        # Format messages for the API
        formatted_messages = self._format_messages(self.channel_history[channel_id])

        response = await self.client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=formatted_messages,
        )

        return response.choices[0].message.content

    async def run_with_text(self, text: str) -> str:
        """
        Run the agent on a single piece of text without any conversation history.
        
        Args:
            text (str): The text to process
            
        Returns:
            str: The agent's response
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]

        response = await self.client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=messages,
        )

        return response.choices[0].message.content

    def yelp_search(self, term: str, zipcode: str) -> tuple:
        """
        Search Yelp for businesses based on search term and zipcode.
        
        Args:
            term (str): Search term (e.g., "pizza", "coffee", "gym")
            zipcode (str): 5-digit US zipcode
            
        Returns:
            tuple: (formatted_string, raw_businesses)
        """
        # Validate zipcode format
        if not re.match(r'^\d{5}$', zipcode):
            return "Error: Please provide a valid 5-digit zipcode.", []

        try:
            # Search Yelp API
            search_response = self.yelp_client.search_query(
                term=term,
                location=zipcode,
                limit=10,  # Get top 10 results
                sort_by='rating'  # Sort by rating
            )

            # Format results
            if not search_response.get('businesses'):
                return f"No results found for '{term}' in zipcode {zipcode}.", []

            results = [f"🔍 Top results for '{term}' in {zipcode}:"]
            
            # Store business details for easy reference
            business_details = {}
            
            # Get detailed information for each business
            for i, business in enumerate(search_response['businesses'], 1):
                try:
                    # Get detailed business information
                    business_id = business['id']
                    details = self.yelp_client.business_query(business_id)
                    
                    # Store business details for reference
                    business_details[str(i)] = {
                        'name': business['name'],
                        'phone': business.get('display_phone', 'N/A'),
                        'address': ', '.join(business['location']['display_address']),
                        'rating': business['rating'],
                        'reviews': business['review_count'],
                        'yelp_url': business.get('url', ''),
                        'website': details.get('url', ''),
                        'hours': details.get('hours', []),
                        'transactions': details.get('transactions', []),
                        'messaging': {
                            'has_messaging': details.get('messaging', {}).get('use_case_text', ''),
                            'response_rate': details.get('messaging', {}).get('response_rate_description', '')
                        }
                    }
                    
                    # Format basic information
                    result_text = [
                        f"\n[{i}] {business['name']}"
                        f"\n    📞 {business.get('display_phone', 'N/A')}"
                        f"\n    ⭐ {business['rating']} ({business['review_count']} reviews)"
                        f"\n    📍 {', '.join(business['location']['display_address'])}"
                    ]
                    
                    # Add price and status in one line
                    status_parts = []
                    if 'price' in details:
                        status_parts.append(details['price'])
                    if 'hours' in details and details['hours']:
                        is_open = details['hours'][0].get('is_open_now', False)
                        status_parts.append('Open' if is_open else 'Closed')
                        
                        # Add business hours if available
                        if details['hours'][0].get('open', []):
                            today_hours = next((hours for hours in details['hours'][0]['open'] 
                                             if hours['day'] == datetime.now().weekday()), None)
                            if today_hours:
                                start = f"{int(today_hours['start'][:2]):02d}:{today_hours['start'][2:]}"
                                end = f"{int(today_hours['end'][:2]):02d}:{today_hours['end'][2:]}"
                                status_parts.append(f"{start}-{end}")
                    
                    if status_parts:
                        result_text.append(f"    💫 {' • '.join(status_parts)}")
                    
                    # Add categories if available
                    if 'categories' in details:
                        categories = [cat['title'] for cat in details['categories']][:3]  # Limit to 3 categories
                        if categories:
                            result_text.append(f"    🏷️ {', '.join(categories)}")
                    
                    # Add transaction types if available
                    if details.get('transactions'):
                        transactions = [t.replace('_', ' ').title() for t in details['transactions']]
                        if transactions:
                            result_text.append(f"    💳 {', '.join(transactions)}")
                    
                    # Add Yelp URL
                    if business.get('url'):
                        result_text.append(f"    🔗 {business['url']}")
                    
                    # Add business website if available
                    if details.get('url'):
                        result_text.append(f"    🌐 {details['url']}")
                    
                    results.append("\n".join(result_text))
                    
                except Exception as detail_error:
                    # Fallback to basic information if detailed query fails
                    results.append(
                        f"\n[{i}] {business['name']}"
                        f"\n    ⭐ {business['rating']} ({business['review_count']} reviews)"
                        f"\n    📍 {', '.join(business['location']['display_address'])}"
                        f"\n    🔗 {business.get('url', 'N/A')}"
                    )

            return "\n".join(results), search_response['businesses']

        except Exception as e:
            return f"Error searching Yelp: {str(e)}", []