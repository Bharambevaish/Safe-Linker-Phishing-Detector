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

def normalize_url(url):
    """
    Lower-cases, trims and adds a default scheme when one is missing.
    Checking for '://' (instead of urlparse().scheme) avoids treating
    'example.com:8080/login' as having the scheme 'example.com'.
    """
    url = str(url).strip().lower()
    if not url or url == "nan":
        return ""
    if "://" not in url:
        url = "http://" + url
    return url


def _host_matches(hostname, domain):
    """True if hostname is exactly `domain` or a subdomain of it."""
    return hostname == domain or hostname.endswith("." + domain)


def extract_features(url):
    """
    Extracts advanced features from a URL for phishing detection.
    Returns a dictionary of feature values.
    """
    try:
        url = normalize_url(url)
        if not url:
            return {}
        parsed = urlparse(url)
        # hostname strips credentials ("user@") and port (":8080") from netloc
        hostname = parsed.hostname or ""
        if not hostname:
            return {}

        features = {}

        # 1. Base Lengths Features
        features['length_url'] = len(url)
        features['length_hostname'] = len(hostname)

        # 2. IP in Domain
        try:
            ipaddress.ip_address(hostname)
            features['ip'] = 1
        except ValueError:
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
        
        # ===============================================
        # ADVANCED FEATURE ENGINEERING (Siemens Upgrade)
        # ===============================================
        
        # 4. URL Shortener Detection
        # Checks if the domain matches popular shortener services.
        # Exact/subdomain match only: a plain substring test flagged hosts like
        # "microsoft.com" because they contain "t.co".
        features['is_shortener'] = 1 if any(_host_matches(hostname, s) for s in SHORTENERS) else 0

        # 5. Prefix/Suffix Check
        # Legitimate sites usually don't have hyphens in the domain, phishers often use them.
        features['domain_has_hyphen'] = 1 if '-' in hostname else 0

        # 6. Subdomain Count
        # More than 2 subdomains is suspicious (excluding a leading www)
        # e.g., www.login.paypal.secure.com -> 2 extra subdomains
        if features['ip']:
            features['subdomain_count'] = 0
        else:
            domain_parts = hostname.removeprefix('www.').split('.')
            features['subdomain_count'] = len(domain_parts) - 2 if len(domain_parts) > 2 else 0

        # 7. Abnormal URL
        # Checks if hostname is part of the URL path (common in open redirects or phishing)
        features['abnormal_url'] = 1 if hostname in parsed.path else 0
        
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
# EXTERNAL THREAT INTELLIGENCE
# ===============================================

SAFE_BROWSING_ENDPOINT = "https://safebrowsing.googleapis.com/v4/threatMatches:find"

SUSPICIOUS_KEYWORDS = ['login', 'signin', 'secure', 'verify', 'update', 'banking', 'account', 'auth', 'password']

# Brand name -> the brand's real domains. A brand name appearing on any other
# host is a classic impersonation signal.
BRAND_DOMAINS = {
    'paypal': ['paypal.com', 'paypal.me'],
    'appleid': ['apple.com'],
    'apple': ['apple.com', 'icloud.com'],
    'google': ['google.com', 'google.co.in', 'gmail.com', 'youtube.com'],
    'microsoft': ['microsoft.com', 'live.com', 'office.com', 'outlook.com'],
    'amazon': ['amazon.com', 'amazon.in'],
    'facebook': ['facebook.com', 'fb.com'],
    'netflix': ['netflix.com'],
    'sbi': ['sbi.co.in', 'onlinesbi.sbi'],
    'hdfc': ['hdfcbank.com'],
    'icici': ['icicibank.com'],
}


def _official_host(hostname):
    return any(_host_matches(hostname, d) for domains in BRAND_DOMAINS.values() for d in domains)


def _google_safe_browsing(url, api_key, timeout=5):
    """
    Real Google Safe Browsing v4 lookup.
    Returns (score, reasons) on success, or None if the call fails so the
    caller can fall back to the offline heuristic.
    """
    import requests  # imported lazily so features.py works without it during training

    body = {
        "client": {"clientId": "safelinker", "clientVersion": "2.1"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE",
                            "POTENTIALLY_HARMFUL_APPLICATION"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }
    try:
        resp = requests.post(SAFE_BROWSING_ENDPOINT, params={"key": api_key}, json=body, timeout=timeout)
        resp.raise_for_status()
        matches = resp.json().get("matches", [])
    except Exception:
        return None

    if not matches:
        return 0.0, []
    types = sorted({m.get("threatType", "UNKNOWN").replace("_", " ").title() for m in matches})
    return 1.0, [f"Threat Intel: Google Safe Browsing lists this URL ({', '.join(types)})."]


def _heuristic_intel(url):
    """Offline stand-in used when no API key is configured (demo mode)."""
    reasons = []
    score = 0.0
    hostname = urlparse(url).hostname or ""
    official = _official_host(hostname)

    if not official and any(k in url for k in SUSPICIOUS_KEYWORDS):
        score = 0.8
        reasons.append("Threat Intel (offline heuristic): credential-related keywords on an unverified domain.")

    impersonated = [b for b in BRAND_DOMAINS if b in url and not official]
    if impersonated:
        score = 1.0
        reasons.append(f"Threat Intel (offline heuristic): mentions '{impersonated[0]}' but is not hosted on its official domain.")

    return score, reasons


def check_external_intel(url, google_api_key=None, phishtank_key=None):
    """
    Returns (score, reasons, mode) where score is 0.0 (clean) to 1.0 (flagged)
    and mode is "live" (Google Safe Browsing answered) or "offline".
    """
    url = normalize_url(url)
    if not url:
        return 0.0, [], "offline"

    if google_api_key:
        result = _google_safe_browsing(url, google_api_key)
        if result is not None:
            return result[0], result[1], "live"

    score, reasons = _heuristic_intel(url)
    return score, reasons, "offline"
