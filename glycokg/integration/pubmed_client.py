#!/usr/bin/env python3
"""
Real PubMed API Client for Literature Integration
Fetches actual literature data from NCBI PubMed database
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import aiohttp
import xml.etree.ElementTree as ET
from urllib.parse import urlencode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PubMedArticle:
    """Data class for PubMed article information"""
    pmid: str
    title: str
    abstract: Optional[str] = None
    authors: List[str] = None
    journal: Optional[str] = None
    publication_date: Optional[str] = None
    doi: Optional[str] = None
    keywords: List[str] = None
    mesh_terms: List[str] = None
    publication_types: List[str] = None
    affiliations: List[str] = None


class RealPubMedClient:
    """
    Real PubMed API client using NCBI E-utilities
    No API key required for basic usage (rate-limited)
    """
    
    def __init__(self, email: str = "research@glycoinformatics.ai"):
        """
        Initialize PubMed client.
        
        Args:
            email: Contact email (required by NCBI)
        """
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.email = email
        self.tool = "glycoinformatics_ai"
        
        # Rate limiting (NCBI allows 3 requests/second without API key)
        self.rate_limit_delay = 0.34  # ~3 requests per second
        self.last_request_time = 0
        
    async def _rate_limit(self):
        """Ensure we don't exceed NCBI rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    async def search_articles(self, 
                            query: str, 
                            max_results: int = 100,
                            sort_order: str = "relevance") -> List[str]:
        """
        Search PubMed for articles and return PMIDs.
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            sort_order: Sort order (relevance, pub_date, etc.)
            
        Returns:
            List of PMIDs
        """
        await self._rate_limit()
        
        # ESearch parameters
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'sort': sort_order,
            'email': self.email,
            'tool': self.tool
        }
        
        search_url = f"{self.base_url}/esearch.fcgi?" + urlencode(params)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url) as response:
                if response.status == 200:
                    xml_content = await response.text()
                    return self._parse_search_results(xml_content)
                else:
                    logger.error(f"PubMed search failed: {response.status}")
                    return []
    
    def _parse_search_results(self, xml_content: str) -> List[str]:
        """Parse PMIDs from search results XML"""
        try:
            root = ET.fromstring(xml_content)
            pmids = []
            
            for id_elem in root.findall('.//Id'):
                pmids.append(id_elem.text)
                
            return pmids
        except ET.ParseError as e:
            logger.error(f"Error parsing search results: {e}")
            return []
    
    async def fetch_articles(self, pmids: List[str]) -> List[PubMedArticle]:
        """
        Fetch detailed article information for given PMIDs.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of PubMedArticle objects
        """
        if not pmids:
            return []
            
        await self._rate_limit()
        
        # EFetch parameters
        pmid_string = ','.join(pmids)
        params = {
            'db': 'pubmed',
            'id': pmid_string,
            'retmode': 'xml',
            'email': self.email,
            'tool': self.tool
        }
        
        fetch_url = f"{self.base_url}/efetch.fcgi?" + urlencode(params)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(fetch_url) as response:
                if response.status == 200:
                    xml_content = await response.text()
                    return self._parse_articles(xml_content)
                else:
                    logger.error(f"PubMed fetch failed: {response.status}")
                    return []
    
    def _parse_articles(self, xml_content: str) -> List[PubMedArticle]:
        """Parse article details from PubMed XML"""
        try:
            root = ET.fromstring(xml_content)
            articles = []
            
            for article_elem in root.findall('.//PubmedArticle'):
                article = self._extract_article_data(article_elem)
                if article:
                    articles.append(article)
                    
            return articles
        except ET.ParseError as e:
            logger.error(f"Error parsing articles: {e}")
            return []
    
    def _extract_article_data(self, article_elem) -> Optional[PubMedArticle]:
        """Extract data from a single article XML element"""
        try:
            # PMID
            pmid_elem = article_elem.find('.//PMID')
            if pmid_elem is None:
                return None
            pmid = pmid_elem.text
            
            # Title
            title_elem = article_elem.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else "No title"
            
            # Abstract
            abstract_elem = article_elem.find('.//Abstract/AbstractText')
            abstract = abstract_elem.text if abstract_elem is not None else None
            
            # Authors
            authors = []
            for author_elem in article_elem.findall('.//Author'):
                lastname_elem = author_elem.find('LastName')
                firstname_elem = author_elem.find('ForeName')
                
                if lastname_elem is not None:
                    lastname = lastname_elem.text
                    firstname = firstname_elem.text if firstname_elem is not None else ""
                    authors.append(f"{firstname} {lastname}".strip())
            
            # Journal
            journal_elem = article_elem.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else None
            
            # Publication date
            pub_date_elem = article_elem.find('.//PubDate')
            pub_date = None
            if pub_date_elem is not None:
                year_elem = pub_date_elem.find('Year')
                month_elem = pub_date_elem.find('Month') 
                day_elem = pub_date_elem.find('Day')
                
                if year_elem is not None:
                    year = year_elem.text
                    month = month_elem.text if month_elem is not None else "01"
                    day = day_elem.text if day_elem is not None else "01"
                    pub_date = f"{year}-{month}-{day}"
            
            # DOI
            doi = None
            for id_elem in article_elem.findall('.//ELocationID'):
                if id_elem.get('EIdType') == 'doi':
                    doi = id_elem.text
                    break
            
            # MeSH terms
            mesh_terms = []
            for mesh_elem in article_elem.findall('.//MeshHeading/DescriptorName'):
                mesh_terms.append(mesh_elem.text)
            
            # Keywords
            keywords = []
            for keyword_elem in article_elem.findall('.//KeywordList/Keyword'):
                keywords.append(keyword_elem.text)
            
            return PubMedArticle(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                publication_date=pub_date,
                doi=doi,
                keywords=keywords,
                mesh_terms=mesh_terms
            )
            
        except Exception as e:
            logger.error(f"Error extracting article data: {e}")
            return None
    
    async def search_glycan_literature(self, 
                                     glytoucan_id: str = None,
                                     max_results: int = 50) -> List[PubMedArticle]:
        """
        Search for literature related to specific glycans or general glycomics.
        
        Args:
            glytoucan_id: Specific GlyTouCan ID to search for
            max_results: Maximum number of articles
            
        Returns:
            List of relevant PubMedArticle objects
        """
        # Build search query
        if glytoucan_id:
            query = f'("{glytoucan_id}" OR glycan OR glycosylation) AND (structure OR mass spectrometry)'
        else:
            query = (
                '(glycan OR glycosylation OR oligosaccharide OR "N-glycan" OR "O-glycan" OR '
                'glycoprotein OR lectin OR carbohydrate) AND '
                '("mass spectrometry" OR proteomics OR glycomics OR structure)'
            )
        
        # Add date filter for recent articles
        query += ' AND ("2020"[Date - Publication] : "3000"[Date - Publication])'
        
        logger.info(f"Searching PubMed with query: {query}")
        
        # Search for PMIDs
        pmids = await self.search_articles(query, max_results)
        logger.info(f"Found {len(pmids)} relevant PMIDs")
        
        if not pmids:
            return []
        
        # Fetch article details in batches (NCBI recommends ≤200 at a time)
        articles = []
        batch_size = 50
        
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            logger.info(f"Fetching batch {i//batch_size + 1}: {len(batch_pmids)} articles")
            
            batch_articles = await self.fetch_articles(batch_pmids)
            articles.extend(batch_articles)
            
            # Rate limiting between batches
            if i + batch_size < len(pmids):
                await asyncio.sleep(1)
        
        logger.info(f"Successfully retrieved {len(articles)} complete articles")
        return articles
    
    def get_pmid_from_doi(self, doi: str) -> Optional[str]:
        """
        Convert DOI to PMID (synchronous helper function).
        In practice, you'd use the NCBI API for this.
        """
        # This would require another API call to NCBI
        # For now, return None
        return None


async def test_pubmed_client():
    """Test the PubMed client with glycan-related searches"""
    
    client = RealPubMedClient()
    
    # Test general glycomics search
    logger.info("Testing general glycomics literature search...")
    articles = await client.search_glycan_literature(max_results=10)
    
    if articles:
        logger.info(f"Retrieved {len(articles)} articles")
        for i, article in enumerate(articles[:3]):
            logger.info(f"Article {i+1}:")
            logger.info(f"  PMID: {article.pmid}")
            logger.info(f"  Title: {article.title[:100]}...")
            logger.info(f"  Authors: {', '.join(article.authors[:3]) if article.authors else 'Unknown'}")
            logger.info(f"  Journal: {article.journal}")
            logger.info("")
    else:
        logger.warning("No articles retrieved")


if __name__ == "__main__":
    asyncio.run(test_pubmed_client())