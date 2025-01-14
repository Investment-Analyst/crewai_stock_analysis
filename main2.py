from crewai import Crew
from textwrap import dedent

from stock_analysis_agents import StockAnalysisAgents
from stock_analysis_tasks import StockAnalysisTasks

from dotenv import load_dotenv
dotenv_path = '.env.example'
load_dotenv(dotenv_path)

class FinancialCrew:
    def __init__(self, company):
        self.company = company

    def run(self):
        agents = StockAnalysisAgents()
        tasks = StockAnalysisTasks()

        # Initialize agents
        research_analyst_agent = agents.research_analyst()
        financial_analyst_agent = agents.financial_analyst()
        investment_advisor_agent = agents.investment_advisor()

        # Execute tasks sequentially with error handling
        try:
            research_result = tasks.research(research_analyst_agent, self.company)
        except Exception as e:
            research_result = f"Error generating research analyst report: {str(e)}"

        try:
            financial_analysis_result = tasks.financial_analysis(financial_analyst_agent)
        except Exception as e:
            financial_analysis_result = f"Error generating financial analyst report: {str(e)}"

        try:
            filings_analysis_result = tasks.filings_analysis(financial_analyst_agent)
        except Exception as e:
            filings_analysis_result = f"Error generating filings analysis: {str(e)}"

        try:
            recommendation_result = tasks.recommend(investment_advisor_agent)
        except Exception as e:
            recommendation_result = f"Error generating investment advisor recommendation: {str(e)}"

        # Combine results into a single comprehensive report
        comprehensive_report = f"""
        股票研究分析師報告:
        {str(research_result)}

        財務分析師報告:
        {str(financial_analysis_result)}

        財務文件分析:
        {str(filings_analysis_result)}

        投資顧問建議:
        {str(recommendation_result)}
        """

        return comprehensive_report

if __name__ == "__main__":
    print("## 歡迎使用股票分析機器人")
    print('-------------------------------')
    company = input(
        dedent("""
          你想分析的股票是什麼？
        """))

    # Run financial crew
    financial_crew = FinancialCrew(company)
    result = financial_crew.run()

    # Display the final report
    print("\n\n########################")
    print("## 這是完整的分析報告")
    print("########################\n")
    print(result)
