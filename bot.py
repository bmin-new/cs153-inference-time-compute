import os
import discord
import logging
import re
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import urllib.parse
import json

from discord.ext import commands
from dotenv import load_dotenv
from agent import MistralAgent

PREFIX = "!"

# Setup logging
logger = logging.getLogger("discord")

# Load the environment variables
load_dotenv()

# Create the bot with all intents
# The message content and members intent must be enabled in the Discord Developer Portal for the bot to work.
intents = discord.Intents.all()
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

# Import the Mistral agent from the agent.py file
agent = MistralAgent()

# Get the token from the environment variables
token = os.getenv("DISCORD_TOKEN")

# Add a global flag at the top of the file
interactive_session_active = False

# Store collected information globally
user_context = {}

@bot.event
async def on_ready():
    """
    Called when the client is done preparing the data received from Discord.
    Prints message on terminal when bot successfully connects to discord.

    https://discordpy.readthedocs.io/en/latest/api.html#discord.on_ready
    """
    logger.info(f"{bot.user} has connected to Discord!")

async def send_long_message(channel, content):
    """Helper function to send long messages by breaking them into chunks."""
    # Split content into chunks of 1900 characters (leaving room for formatting)
    chunks = [content[i:i+1900] for i in range(0, len(content), 1900)]
    
    # Send each chunk as a separate message
    for chunk in chunks:
        await channel.send(chunk)

@bot.event
async def on_message(message: discord.Message):
    """
    Called when a message is sent in any channel the bot can see.

    https://discordpy.readthedocs.io/en/latest/api.html#discord.on_message
    """
    global interactive_session_active

    # Don't delete this line! It's necessary for the bot to process commands.
    await bot.process_commands(message)

    # Ignore messages from self or other bots to prevent infinite loops.
    # Also ignore messages that start with the command prefix
    if message.author.bot or message.content.startswith(PREFIX) or interactive_session_active:
        return

    # Get the channel's last message before this one
    async for last_msg in message.channel.history(limit=1, before=message):
        if last_msg.author == bot.user and (
            "What type of business or service are you looking for?" in last_msg.content or
            "Please enter your 5-digit zip code:" in last_msg.content
        ):
            return
        break

    # Process the message with the agent
    logger.info(f"Processing message from {message.author}: {message.content}")
    
    response = await agent.run(message)

    # Send the response using the chunking helper
    await send_long_message(message.channel, response)

# Commands

# This example command is here to show you how to add commands to the bot.
# Run !ping with any number of arguments to see the command in action.
# Feel free to delete this if your project will not need commands.
@bot.command(name="ping", help="Pings the bot.")
async def ping(ctx, *, arg=None):
    if arg is None:
        await ctx.send("Pong!")
    else:
        await ctx.send(f"Pong! Your argument was {arg}")

@bot.command(name="ask", help="Ask the agent a question without conversation history.")
async def ask(ctx, *, arg=None):
    """
    Ask the agent a question without using conversation history.
    Usage: !ask What is the capital of France?
    """
    if arg is None:
        await ctx.send("Please provide a question to ask the agent. Usage: !ask What is the capital of France?")
        return
    
    try:
        response = await agent.run_with_text(arg)
        await ctx.send(response)
    except Exception as e:
        logger.error(f"Error in ask command: {e}")
        await ctx.send("Sorry, I encountered an error while processing your question.")

@bot.command(name="search", help="Search for businesses on Yelp.")
async def search(ctx, term: str, zipcode: str):
    """
    Search for businesses on Yelp.
    Usage: !search "pizza" 90210
    """
    try:
        # Send initial message to show we're working
        status_message = await ctx.send(f"🔍 Searching for '{term}' in {zipcode}...")
        
        # Get search results
        response, businesses = agent.yelp_search(term, zipcode)
        
        # Split and send search results
        chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
        for chunk in chunks:
            await ctx.send(chunk)
            
        # Clean up status message
        await status_message.delete()
            
    except Exception as e:
        logger.error(f"Error in search command: {e}")
        await ctx.send("Sorry, I encountered an error while searching.")

async def get_dynamic_questions(agent: MistralAgent, business_type: str) -> list:
    """Get dynamically generated questions based on business type using the Mistral agent."""
    prompt = f"""Given a user is looking for a {business_type}, generate a list of 4-6 relevant questions that would help gather information to request a quote or service details.
    The questions should be specific to this type of business/service and help draft a detailed message to the business.
    Format each question on a new line starting with a number and a period (e.g., "1. Question text").
    Consider what information would be most relevant for this specific type of business to provide an accurate quote or service details."""
    
    try:
        response = await agent.run_with_text(prompt)
        
        # Extract questions from response
        questions = []
        for line in response.split('\n'):
            # Look for lines that start with a number and period
            if re.match(r'^\d+\.', line.strip()):
                # Remove the number and leading/trailing whitespace
                question = re.sub(r'^\d+\.\s*', '', line.strip())
                if question:
                    questions.append(question)
        
        return questions if questions else [
            "What specific services do you need?",
            "When do you need this service?",
            "Do you have any specific requirements or preferences?",
            "What is your budget range?",
            "Is there anything else the business should know?"
        ]
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        # Return default questions if there's an error
        return [
            "What specific services do you need?",
            "When do you need this service?",
            "Do you have any specific requirements or preferences?",
            "What is your budget range?",
            "Is there anything else the business should know?"
        ]

async def draft_dynamic_message(agent: MistralAgent, business_type: str, answers: dict) -> str:
    """Draft a personalized message based on business type and answers using the Mistral agent."""
    # Create a prompt that includes all the information
    prompt = f"""Draft a professional message to a {business_type} business requesting a quote or service information.
    Use the following information to create the message:

    Business Type: {business_type}
    Customer Responses:
    {json.dumps(answers, indent=2)}

    The message should:
    1. Be professional and courteous
    2. Include all relevant information from the customer's responses, but NEVER disclose the user's budget upfront.
    3. Be written by a master negotiator aiming to get the best possible offer.
    4. Request pricing and availability information clearly.
    5. End with a strong call to action.
    6. Always conclude with:\n\nBest,\n[Your Name]\n[Your Contact Information]

    Format the message as a plain text email without any markdown or special formatting."""

    try:
        message = await agent.run_with_text(prompt)
        return message.strip()
    except Exception as e:
        logger.error(f"Error drafting message: {e}")
        # Return a basic message if there's an error
        return f"""Hi! I found your business on Yelp and I'm interested in your services.

I'm looking for {business_type} services and would like to request a quote. Here are my requirements:

{chr(10).join(f'- {q}: {a}' for q, a in answers.items())}

Could you please provide information about your services, pricing, and availability? Thank you!

Best,\n[Your Name]\n[Your Contact Information]"""

@bot.command(name="get", help="Collect business type, zip code, and follow-up questions")
async def get(ctx):
    global interactive_session_active
    interactive_session_active = True

    try:
        await ctx.send("\n🔍 What type of business or service are you looking for? (e.g., pizza, dentist, plumber, movers, tax services, etc.)")

        def check(message):
            return message.author == ctx.author and message.channel == ctx.channel

        business_response = await bot.wait_for('message', timeout=30.0, check=check)
        business_type = business_response.content.strip()

        if len(business_type) < 2:
            await ctx.send("❌ Please provide a valid business type (at least 2 characters).")
            return

        await ctx.send("📍 Please enter your 5-digit zip code:")
        zip_response = await bot.wait_for('message', timeout=30.0, check=check)
        zipcode = zip_response.content.strip()

        if not re.match(r'^\d{5}$', zipcode):
            await ctx.send("❌ Please provide a valid 5-digit zip code.")
            return

        await ctx.send("🔄 Generating relevant questions based on your request...\n📝 I'll ask you a few questions to gather more context.")

        questions = await get_dynamic_questions(agent, business_type)
        answers = {}

        for i, question in enumerate(questions, 1):
            await ctx.send(f"🔹 Question {i}/{len(questions)}: {question}")
            try:
                answer = await bot.wait_for('message', timeout=60.0, check=lambda m: m.author == ctx.author and m.channel == ctx.channel)
                answers[question] = answer.content.strip()
            except asyncio.TimeoutError:
                await ctx.send("❌ You took too long to respond. Please try again with !get")
                return

        await ctx.send("✅ Information collected successfully! You can now use !list to see Yelp results.")

        user_context[ctx.author.id] = {'business_type': business_type, 'zipcode': zipcode, 'answers': answers}

    except Exception as e:
        logger.error(f"Error in get command: {e}")
        await ctx.send("Sorry, I encountered an error while processing your request.")
    finally:
        interactive_session_active = False

@bot.command(name="shutdown", help="Shutdown the bot (owner only)")
@commands.is_owner()
async def shutdown(ctx):
    """Shutdown the bot. Only the bot owner can use this."""
    logger.info(f"Shutdown command received from {ctx.author}")
    await ctx.send("Shutting down...")
    await bot.close()

@bot.command(name="message", help="Send a message to a business through Yelp")
async def message_business(ctx, business_number: str, *, message: str = None):
    """
    Send a message to a business through Yelp's messaging system.
    Usage: !message 1 "I'm interested in getting a quote for catering 50 people next week"
    """
    try:
        # Get Yelp API key
        YELP_API_KEY = os.getenv("YELP_API_KEY")
        if not YELP_API_KEY:
            await ctx.send("❌ Yelp API key not configured. Please set the YELP_API_KEY environment variable.")
            return

        # Get the last search results
        search_results = None
        async for msg in ctx.channel.history(limit=20):
            if msg.author == bot.user and "🔍 Top results for" in msg.content:
                search_results = msg
                break
        
        if not search_results:
            await ctx.send("❌ Please run a !search command first to get business results.")
            return
            
        # Find the business details
        business_info = None
        lines = search_results.content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith(f"[{business_number}]"):
                business_info = {
                    'name': line.replace(f"[{business_number}] ", ""),
                    'yelp_url': None
                }
                # Look for Yelp URL
                for j in range(i, min(i+7, len(lines))):
                    if "    🔗 " in lines[j]:
                        business_info['yelp_url'] = lines[j].replace("    🔗 ", "").strip()
                        break
                break
        
        if not business_info or not business_info['yelp_url']:
            await ctx.send(f"❌ Could not find Yelp page for business number {business_number}.")
            return

        # Create default message if none provided
        if not message:
            message = "Hi! I found your business on Yelp and I'm interested in getting more information about your services. Could you please provide details about pricing and availability? Thank you!"

        # Extract business ID from Yelp URL
        business_id = business_info['yelp_url'].split('biz/')[-1].split('?')[0]

        # Create message URL
        message_url = f"https://www.yelp.com/message_the_business/{business_id}"
        
        # Send instructions to the user
        instructions = [
            f"📨 To message {business_info['name']} through Yelp:",
            "",
            "1. Click this link to open Yelp's messaging page:",
            message_url,
            "",
            "2. Copy and paste this message (or write your own):",
            "```",
            message,
            "```",
            "",
            "Note: You'll need to be logged into your Yelp account to send the message."
        ]
        
        await send_long_message(ctx.channel, "\n".join(instructions))
        
    except Exception as e:
        logger.error(f"Error in message command: {e}")
        await ctx.send("Sorry, I encountered an error while trying to set up the Yelp message.")

@bot.command(name="list", help="List top 10 Yelp businesses based on collected info")
async def list_businesses(ctx):
    global user_context

    if not user_context.get(ctx.author.id):
        await ctx.send("❌ Please run !get first to provide your business type and zip code.")
        return

    business_type = user_context[ctx.author.id]['business_type']
    zipcode = user_context[ctx.author.id]['zipcode']

    confirm_msg = await ctx.send(f"🔍 Searching for **{business_type}** in zip code **{zipcode}**...")
    response, businesses = agent.yelp_search(business_type, zipcode)
    await confirm_msg.delete()

    # Explicitly remove the secondary website link (🌐)
    response = re.sub(r'\s+🌐 .*', '', response)

    chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
    for chunk in chunks:
        await ctx.send(chunk)

@bot.command(name="initiate", help="Collect info, list Yelp businesses, draft messages, and provide Yelp page links")
async def initiate(ctx):
    await ctx.send("✨✨✨ **__Welcome to Yelp Outreach Assistant!__** ✨✨✨\nLet's find the best businesses and craft personalized messages for you!\n\nHere's how it works:\n1. Provide your business type and zip code.\n2. Answer a few quick questions to help us understand your needs.\n3. We'll find the top businesses on Yelp for you.\n4. You'll receive personalized messages ready to send!\n\n⬇️")
    await ctx.invoke(bot.get_command('get'))
    await ctx.invoke(bot.get_command('list'))

    global user_context

    if not user_context.get(ctx.author.id):
        await ctx.send("❌ Please run !get first to provide your business type and zip code.")
        return

    business_type = user_context[ctx.author.id]['business_type']
    zipcode = user_context[ctx.author.id]['zipcode']
    answers = user_context[ctx.author.id].get('answers', {})

    await ctx.send("📝 Drafting personalized outreach messages and providing Yelp business links...")

    _, businesses = agent.yelp_search(business_type, zipcode)

    for i, business in enumerate(businesses, 1):
        personalized_message = await draft_dynamic_message(agent, business_type, {**answers, "Business Name": business['name']})
        yelp_url = business.get('url', 'N/A')

        instructions = (
            f"[{i}] {business['name']}\n{personalized_message}\n\n"
            f"🔗 [Click here to open Yelp page]({business['url']})\n"
            "👉 Click 'Message the Business' on Yelp to send your message.\n"
        )
        await send_long_message(ctx.channel, instructions)

@bot.command(name="draft", help="Draft personalized outreach messages based on collected info")
async def draft(ctx):
    global user_context

    if not user_context.get(ctx.author.id):
        await ctx.send("❌ Please run !get first to provide your business type and zip code.")
        return

    business_type = user_context[ctx.author.id]['business_type']
    zipcode = user_context[ctx.author.id]['zipcode']
    answers = user_context[ctx.author.id].get('answers', {})

    await ctx.send("📝 Drafting personalized outreach messages for each business...")

    _, businesses = agent.yelp_search(business_type, zipcode)

    for i, business in enumerate(businesses, 1):
        personalized_message = await draft_dynamic_message(agent, business_type, {**answers, "Business Name": business['name']})
        yelp_url = business.get('url', 'N/A')

        formatted_message = (
            f"📝 **Outreach for {business['name']}**\n\n⬇️\n\n"
            f"{personalized_message}\n\n"
            f"🔗 [Click here to open Yelp page]({yelp_url})"
        )

        await send_long_message(ctx.channel, formatted_message)

@bot.command(name="send-message", help="Provide instructions to send messages through Yelp")
async def send_message(ctx, business_number: str):
    await ctx.invoke(bot.get_command('message'), business_number=business_number)

@bot.command(name="welcome", help="Show a detailed welcome message explaining bot capabilities")
async def welcome(ctx):
    welcome_message = (
        "✨✨✨ **Welcome to Yelp Outreach Assistant!** ✨✨✨\n\n"
        "I'm here to help you effortlessly find local businesses and automate your outreach process. Whether you're:\n\n"
        "- 📦 **Comparing quotes across movers** for your upcoming relocation\n"
        "- 📊 **Searching for a new tax representative** to handle your finances\n"
        "- 🏢 **Looking for a real estate broker** to find your perfect office space\n\n"
        "I've got you covered!\n\n"
        "Here's how you can get started:\n"
        "1. Use the `!initiate` command to begin.\n"
        "2. Provide your business type and zip code.\n"
        "3. Answer a few quick questions to help me understand your needs.\n"
        "4. I'll find the top businesses on Yelp tailored to your requirements.\n"
        "5. You'll receive personalized outreach messages ready to send!\n\n"
        "Ready to simplify your search and outreach? Just type `!initiate` to begin! 🚀"
    )

    await send_long_message(ctx.channel, welcome_message)

# Start the bot, connecting it to the gateway
if __name__ == "__main__":
    try:
        logger.info("Starting bot...")
        bot.run(token)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Bot stopped due to error: {e}")
    finally:
        logger.info("Bot shutdown complete")