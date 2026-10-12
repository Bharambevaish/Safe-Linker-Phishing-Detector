import re
import ipaddress
from urllib.parse import urlparse

# List of common URL shorteners
SHORTENERS = [
    "bit.ly", "goo.gl", "t.co", "tinyurl.com", "is.gd", "cli.gs",
    "yfrog.com", "migre.me", "ff.im", "twit.ac", "su.pr", "ow.ly",
    "snipr.com", "short.to", "budurl.com", "ping.fm", "post.ly",
    "just.as", "bkite.com", "moourl.com", "snipurl.com", "q.gs", "shorte.st"
]

def extract_features(url):
    try:
        url = str(url).strip().lower()
        parsed = urlparse(url)
        
        # Add a default scheme if missing for proper parsing
        if not parsed.scheme:
            url = 'http://' + url
            parsed = urlparse(url)
            
        features = {}
        
        # 1. Base Lengths Features
        features['length_url'] = len(url)
        features['length_hostname'] = len(parsed.netloc)
        
        # 2. IP in Domain
        try:
            ipaddress.ip_address(parsed.netloc.split(':')[0])
            features['ip'] = 1
        except:
            features['ip'] = 0

        # 3. Basic Special Character Counts
        features['nb_dots'] = url.count('.')
        features['nb_hyphens'] = url.count('-')
        features['nb_at'] = url.count('@')
        features['nb_qm'] = url.count('?')
        features['nb_and'] = url.count('&')
        features['nb_or'] = url.count('|')
        features['nb_eq'] = url.count('=')
        features['nb_underscore'] = url.count('_')
        features['nb_percent'] = url.count('%')
        features['nb_slash'] = url.count('/')
        features['nb_dslash'] = url.count('//') - 1 if url.count('//') > 0 else 0
        features['nb_www'] = url.count('www')
        features['nb_com'] = url.count('.com')
        features['https_token'] = 1 if parsed.scheme == 'https' else 0
        features['http_in_path'] = 1 if 'http' in parsed.path else 0
        
      
        # ADVANCED FEATURE ENGINEERING 
       
        
        # 4. URL Shortener Detection
        # Checks if the domain matches popular shortener services.
        features['is_shortener'] = 1 if any(short in parsed.netloc for short in SHORTENERS) else 0
        
        # 5. Prefix/Suffix Check 
        # Legitimate sites usually don't have hyphens in the domain, phishers often use them.
        features['domain_has_hyphen'] = 1 if '-' in parsed.netloc else 0
        
        # 6. Subdomain Count
        # More than 2 subdomains is suspicious (excluding www)
        # e.g., www.login.paypal.secure.com -> 4 dots
        domain_parts = parsed.netloc.replace('www.', '').split('.')
        features['subdomain_count'] = len(domain_parts) - 2 if len(domain_parts) > 2 else 0
        
        # 7. Abnormal URL
        # Checks if hostname is part of the URL path (common in open redirects or phishing)
        features['abnormal_url'] = 1 if parsed.netloc != "" and parsed.netloc in parsed.path else 0
        
        # Additional features requested by Original Codebase to map smoothly safely
        features['nb_tilde'] = url.count('~')
        features['nb_star'] = url.count('*')
        features['nb_colon'] = url.count(':')
        features['nb_comma'] = url.count(',')
        features['nb_semicolumn'] = url.count(';')
        features['nb_dollar'] = url.count('$')
        features['nb_space'] = url.count(' ')
        
        return features

    except Exception as e:
        # Fallback values if parsing entirely fails
        return {}

def rule_based_severity(features_dict):
    """
    Evaluates rule-based features to give a rule-based severity score (0 to 1).
    Used in hybridization. Also tracks reasons for Siemens 'Deep Dive'.
    """
    score = 0
    reasons = []
    
    if features_dict.get('is_shortener', 0) == 1:
        score += 0.3
        reasons.append("High Risk: Uses a known URL shortener service, a common tactic for hiding malicious payloads.")
    
    if features_dict.get('ip', 0) == 1:
        score += 0.4
        reasons.append("High Risk: Domain is an IP address instead of a recognized hostname.")
        
    if features_dict.get('domain_has_hyphen', 0) == 1:
        score += 0.15
        reasons.append("Warning: Domain uses hyphens (Prefix/Suffix technique) which is common in spoofing.")
        
    if features_dict.get('subdomain_count', 0) > 1:
        score += 0.15
        reasons.append(f"Warning: Excessive subdomains ({features_dict.get('subdomain_count')}) often used to trick users.")
        
    if features_dict.get('length_url', 0) > 75:
        score += 0.1
        reasons.append("Warning: URL is abnormally long, often used to obscure malicious intent.")
        
    if features_dict.get('abnormal_url', 0) == 1:
        score += 0.2
        reasons.append("High Risk: Hostname detected inside the path string (Abnormal URL).")
        
    # Cap score at 1.0 (100% rule severity)
    return min(score, 1.0), reasons

# ===============================================
# EXTERNAL INTELLIGENCE MOcK
# ===============================================

def check_external_intel(url, google_api_key=None, phishtank_key=None):
    """
    # MOCK API: Checks Google Safe Browsing and PhishTank.
    # Replace the mocked response below with the actual `requests` call when keys are provided.
    """
    reasons = []
    intel_score = 0.0 # 0.0 means safe, 1.0 means flagged
    
    # -------------------------------------------------------------
    # TODO for real implementation: 
    # Use requests.post for Google Safe Browsing V4 API 
    # Use requests.get for Phishtank CheckURL API
    # -------------------------------------------------------------
    
    # MOCK LOGIC for demo purposes
    suspicious_keywords = ['login', 'secure', 'verify', 'update', 'banking', 'account', 'auth']
    if any(k in url for k in suspicious_keywords) and ('google' not in url):
        # Fake a hit on an API
        intel_score = 0.8
        reasons.append("Threat Intel: URL keyword heuristics overlap with flagged databases (Mock PhishTank flag).")
        
    if "paypal" in url or "appleid" in url:
        intel_score = 1.0
        reasons.append("Threat Intel: Flagged by Google Safe Browsing (Mock response) as highly deceptive.")

    return intel_score, reasons
