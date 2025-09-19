# @title Zero Trust Code Guardrails Setup
# @markdown Install required dependencies
!pip install pyhcl2 pyyaml > /dev/null

# @markdown Import necessary libraries
import json
import re
import logging
import yaml
import hcl2
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from datetime import datetime
from IPython.display import display, Markdown, JSON

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# @title Define Enums and Classes
from enum import Enum

class PolicyType(Enum):
    """Enumeration of supported policy types"""
    OPENAPI = "openapi"
    ACP = "acp"
    MCP = "mcp"
    HCL = "hcl"
    UNKNOWN = "unknown"

class ZeroTrustPrinciple(Enum):
    """Zero Trust principles to enforce"""
    LEAST_PRIVILEGE = "least_privilege"
    EXPLICIT_VERIFICATION = "explicit_verification"
    ASSUME_BREACH = "assume_breach"
    MICROSEGMENTATION = "microsegmentation"
    DEVICE_TRUST = "device_trust"
    USER_TRUST = "user_trust"
    DATA_ENCRYPTION = "data_encryption"
    IDENTITY_CENTRIC = "identity_centric"
    DATA_SOVEREIGNTY = "data_sovereignty"
    TEMPORAL_ACCESS = "temporal_access"
    SECRET_MANAGEMENT = "secret_management"
    AUDITING_IMMUTABILITY = "auditing_immutability"

class PolicyViolation:
    """Represents a policy violation found during analysis"""
    
    def __init__(self, policy_type: PolicyType, principle: ZeroTrustPrinciple, 
                 message: str, severity: str, location: str, aspect: str):
        self.policy_type = policy_type
        self.principle = principle
        self.message = message
        self.severity = severity  # "high", "medium", "low"
        self.location = location
        self.aspect = aspect  # Which of the 5 aspects this relates to
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert violation to dictionary for serialization"""
        return {
            "policy_type": self.policy_type.value,
            "principle": self.principle.value,
            "message": self.message,
            "severity": self.severity,
            "location": self.location,
            "aspect": self.aspect
        }
    
    def __str__(self) -> str:
        return f"{self.severity.upper()}: {self.aspect} - {self.message} ({self.location})"

# @title Multi-Modal Guardrail Engine
class MultiModalGuardrailEngine:
    """Main engine for analyzing code against Zero Trust principles"""
    
    def __init__(self, allowed_regions: List[str] = None):
        self.violations: List[PolicyViolation] = []
        self.llm_client = self._initialize_llm_client()
        self.allowed_regions = allowed_regions or ["EU", "europe-west1", "europe-west2", "us-east-1"]
        self.secret_patterns = [
            r'[A-Za-z0-9+/]{40,}',  # Base64-like strings of minimum length
            r'[a-fA-F0-9]{64}',     # SHA256 hash-like strings
            r'[a-fA-F0-9]{128}',    # SHA512 hash-like strings
            r'sk_live_[0-9a-zA-Z]{24,}',  # Stripe secret key pattern
            r'AKIA[0-9A-Z]{16}',    # AWS access key pattern
            r'eyJhbGciOiJ[^"]+',    # JWT token pattern
        ]
        
    def _initialize_llm_client(self) -> Any:
        """
        Initialize the LLM client (placeholder for actual implementation)
        In a real implementation, this would connect to an LLM API
        """
        # This is a placeholder - in practice, you would initialize
        # a connection to your preferred LLM service
        logger.info("Initializing LLM client (placeholder)")
        return None
    
    def analyze_code(self, code: str, policy_type: PolicyType) -> List[PolicyViolation]:
        """
        Analyze code against Zero Trust principles based on policy type
        """
        self.violations = []
        
        try:
            if policy_type == PolicyType.OPENAPI:
                self._analyze_openapi(code)
            elif policy_type == PolicyType.ACP:
                self._analyze_acp(code)
            elif policy_type == PolicyType.MCP:
                self._analyze_mcp(code)
            elif policy_type == PolicyType.HCL:
                self._analyze_hcl(code)
            else:
                logger.error(f"Unsupported policy type: {policy_type}")
                
        except Exception as e:
            logger.error(f"Error analyzing {policy_type.value} code: {str(e)}")
            self.violations.append(
                PolicyViolation(
                    policy_type, 
                    ZeroTrustPrinciple.EXPLICIT_VERIFICATION,
                    f"Failed to parse {policy_type.value} code: {str(e)}",
                    "high",
                    "global",
                    "Parsing"
                )
            )
        
        return self.violations
    
    def _analyze_openapi(self, code: str) -> None:
        """Analyze OpenAPI specification for Zero Trust compliance"""
        try:
            spec = yaml.safe_load(code) if code.strip().startswith('---') else json.loads(code)
            
            # Original checks
            self._check_openapi_auth(spec)
            self._check_openapi_paths(spec)
            self._check_openapi_encryption(spec)
            
            # New aspect checks
            self._check_identity_centric_openapi(spec)
            self._check_data_sovereignty_openapi(spec)
            self._check_secret_management_openapi(code, spec)
            self._check_auditing_openapi(spec)
            
            # Use LLM for more complex analysis
            self._llm_analysis(code, PolicyType.OPENAPI)
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid OpenAPI specification: {str(e)}")
    
    def _check_openapi_auth(self, spec: Dict[str, Any]) -> None:
        """Check OpenAPI spec for authentication requirements"""
        security = spec.get('security', [])
        components = spec.get('components', {}).get('securitySchemes', {})
        
        if not security and not components:
            self.violations.append(
                PolicyViolation(
                    PolicyType.OPENAPI,
                    ZeroTrustPrinciple.EXPLICIT_VERIFICATION,
                    "No authentication security schemes defined",
                    "high",
                    "components.securitySchemes",
                    "Authentication"
                )
            )
    
    def _check_openapi_paths(self, spec: Dict[str, Any]) -> None:
        """Check OpenAPI paths for least privilege violations"""
        paths = spec.get('paths', {})
        
        for path, methods in paths.items():
            for method, details in methods.items():
                if method.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                    # Check for overly permissive operations
                    if not details.get('security'):
                        self.violations.append(
                            PolicyViolation(
                                PolicyType.OPENAPI,
                                ZeroTrustPrinciple.LEAST_PRIVILEGE,
                                f"No security requirements defined for {method.upper()} {path}",
                                "medium",
                                f"paths.{path}.{method}.security",
                                "Least Privilege"
                            )
                        )
    
    def _check_openapi_encryption(self, spec: Dict[str, Any]) -> None:
        """Check OpenAPI spec for encryption requirements"""
        servers = spec.get('servers', [])
        
        for server in servers:
            url = server.get('url', '')
            if url.startswith('http://') and not url.startswith('http://localhost'):
                self.violations.append(
                    PolicyViolation(
                        PolicyType.OPENAPI,
                        ZeroTrustPrinciple.DATA_ENCRYPTION,
                        f"Server URL uses HTTP instead of HTTPS: {url}",
                        "high",
                        "servers",
                        "Data Encryption"
                    )
                )
    
    def _check_identity_centric_openapi(self, spec: Dict[str, Any]) -> None:
        """Check OpenAPI for identity-centric policy binding"""
        components = spec.get('components', {}).get('securitySchemes', {})
        paths = spec.get('paths', {})
        
        # Check if OAuth2 scopes are properly defined and used
        oauth2_schemes = {name: scheme for name, scheme in components.items() 
                         if scheme.get('type') == 'oauth2'}
        
        if oauth2_schemes:
            for path, methods in paths.items():
                for method, details in methods.items():
                    if method.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                        security = details.get('security', [])
                        for sec_req in security:
                            for scheme_name in sec_req.keys():
                                if scheme_name in oauth2_schemes:
                                    # Check if scopes are specified
                                    if not sec_req[scheme_name]:
                                        self.violations.append(
                                            PolicyViolation(
                                                PolicyType.OPENAPI,
                                                ZeroTrustPrinciple.IDENTITY_CENTRIC,
                                                f"OAuth2 scheme {scheme_name} used without required scopes on {method.upper()} {path}",
                                                "medium",
                                                f"paths.{path}.{method}.security",
                                                "Identity-Centric Policy"
                                            )
                                        )
    
    def _check_data_sovereignty_openapi(self, spec: Dict[str, Any]) -> None:
        """Check OpenAPI for data sovereignty compliance"""
        servers = spec.get('servers', [])
        
        for server in servers:
            url = server.get('url', '')
            # Check if server URL points to allowed regions
            if any(region.lower() in url.lower() for region in self.allowed_regions):
                continue
                
            # If we have external URLs not in allowed regions, flag them
            if '://' in url and 'localhost' not in url and '127.0.0.1' not in url:
                self.violations.append(
                    PolicyViolation(
                        PolicyType.OPENAPI,
                        ZeroTrustPrinciple.DATA_SOVEREIGNTY,
                        f"Server URL points to potentially non-compliant region: {url}",
                        "medium",
                        "servers",
                        "Data Sovereignty"
                    )
                )
    
    def _check_secret_management_openapi(self, code: str, spec: Dict[str, Any]) -> None:
        """Check OpenAPI for secret management issues"""
        # Look for potential secrets in the code
        for pattern in self.secret_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                # Avoid flagging common non-secret patterns
                if not self._is_false_positive(match.group()):
                    self.violations.append(
                        PolicyViolation(
                            PolicyType.OPENAPI,
                            ZeroTrustPrinciple.SECRET_MANAGEMENT,
                            f"Potential hardcoded secret detected: {match.group()[:20]}...",
                            "high",
                            "code",
                            "Secret Management"
                        )
                    )
    
    def _check_auditing_openapi(self, spec: Dict[str, Any]) -> None:
        """Check OpenAPI for auditing capabilities"""
        paths = spec.get('paths', {})
        
        # Check if important endpoints have audit-related extensions
        for path, methods in paths.items():
            for method, details in methods.items():
                if method.lower() in ['post', 'put', 'delete', 'patch']:
                    # These methods typically change state and should be audited
                    if not details.get('x-audit', False) and not details.get('x-log', False):
                        self.violations.append(
                            PolicyViolation(
                                PolicyType.OPENAPI,
                                ZeroTrustPrinciple.AUDITING_IMMUTABILITY,
                                f"State-changing operation {method.upper()} {path} lacks explicit audit logging",
                                "medium",
                                f"paths.{path}.{method}",
                                "Auditing & Immutability"
                            )
                        )
    
    def _analyze_hcl(self, code: str) -> None:
        """Analyze HCL configuration for Zero Trust compliance"""
        try:
            config = hcl2.loads(code)
            
            # Original checks
            self._check_hcl_firewall_rules(config)
            self._check_hcl_encryption(config)
            self._check_hcl_microsegmentation(config)
            
            # New aspect checks
            self._check_identity_centric_hcl(config)
            self._check_data_sovereignty_hcl(config)
            self._check_temporal_access_hcl(config)
            self._check_secret_management_hcl(code, config)
            self._check_auditing_immutability_hcl(config)
            
            # Use LLM for more complex analysis
            self._llm_analysis(code, PolicyType.HCL)
            
        except Exception as e:
            raise ValueError(f"Invalid HCL configuration: {str(e)}")
    
    def _check_hcl_firewall_rules(self, config: Dict[str, Any]) -> None:
        """Check HCL firewall rules for least privilege violations"""
        # Extract firewall rules from HCL configuration
        resource_blocks = config.get('resource', [])
        
        for block_type, blocks in resource_blocks.items():
            if 'aws_security_group' in block_type or 'google_compute_firewall' in block_type:
                for block_name, block_config in blocks.items():
                    rules = block_config.get('ingress', []) + block_config.get('egress', [])
                    
                    for rule in rules:
                        # Check for overly permissive rules
                        if rule.get('cidr_blocks', ['0.0.0.0/0']) == ['0.0.0.0/0']:
                            self.violations.append(
                                PolicyViolation(
                                    PolicyType.HCL,
                                    ZeroTrustPrinciple.LEAST_PRIVILEGE,
                                    f"Overly permissive CIDR block in {block_type}.{block_name}",
                                    "high",
                                    f"resource.{block_type}.{block_name}",
                                    "Least Privilege"
                                )
                            )
    
    def _check_hcl_encryption(self, config: Dict[str, Any]) -> None:
        """Check HCL configuration for encryption requirements"""
        # Check for unencrypted resources
        resource_blocks = config.get('resource', [])
        
        for block_type, blocks in resource_blocks.items():
            if 'aws_s3_bucket' in block_type:
                for block_name, block_config in blocks.items():
                    if not block_config.get('server_side_encryption_configuration'):
                        self.violations.append(
                            PolicyViolation(
                                PolicyType.HCL,
                                ZeroTrustPrinciple.DATA_ENCRYPTION,
                                f"S3 bucket without encryption: {block_name}",
                                "high",
                                f"resource.{block_type}.{block_name}",
                                "Data Encryption"
                            )
                        )
    
    def _check_hcl_microsegmentation(self, config: Dict[str, Any]) -> None:
        """Check HCL configuration for microsegmentation"""
        # Check if network is properly segmented
        resource_blocks = config.get('resource', [])
        has_network_segmentation = False
        
        for block_type, blocks in resource_blocks.items():
            if 'aws_vpc' in block_type or 'google_compute_network' in block_type:
                for block_name, block_config in blocks.items():
                    # Check for proper network segmentation
                    if block_config.get('enable_classiclink', False):
                        self.violations.append(
                            PolicyViolation(
                                PolicyType.HCL,
                                ZeroTrustPrinciple.MICROSEGMENTATION,
                                f"ClassicLink enabled in VPC: {block_name}",
                                "medium",
                                f"resource.{block_type}.{block_name}",
                                "Microsegmentation"
                            )
                        )
        
        if not has_network_segmentation:
            self.violations.append(
                PolicyViolation(
                    PolicyType.HCL,
                    ZeroTrustPrinciple.MICROSEGMENTATION,
                    "No explicit network segmentation configured",
                    "medium",
                    "network_configuration",
                    "Microsegmentation"
                )
            )
    
    def _check_identity_centric_hcl(self, config: Dict[str, Any]) -> None:
        """Check HCL for identity-centric policy binding"""
        resource_blocks = config.get('resource', [])
        
        for block_type, blocks in resource_blocks.items():
            if 'aws_iam_policy' in block_type or 'google_project_iam' in block_type:
                for block_name, block_config in blocks.items():
                    policy = block_config.get('policy', '{}')
                    try:
                        policy_json = json.loads(policy)
                        # Check if policy uses resource-based rather than identity-based permissions
                        if self._is_overly_permissive_iam(policy_json):
                            self.violations.append(
                                PolicyViolation(
                                    PolicyType.HCL,
                                    ZeroTrustPrinciple.IDENTITY_CENTRIC,
                                    f"IAM policy {block_name} may be overly permissive or not identity-centric",
                                    "high",
                                    f"resource.{block_type}.{block_name}",
                                    "Identity-Centric Policy"
                                )
                            )
                    except json.JSONDecodeError:
                        # Skip if policy isn't valid JSON
                        continue
    
    def _check_data_sovereignty_hcl(self, config: Dict[str, Any]) -> None:
        """Check HCL for data sovereignty compliance"""
        resource_blocks = config.get('resource', [])
        
        for block_type, blocks in resource_blocks.items():
            if 'aws_s3_bucket' in block_type:
                for block_name, block_config in blocks.items():
                    region = block_config.get('region', '')
                    if region and region not in self.allowed_regions:
                        self.violations.append(
                            PolicyViolation(
                                PolicyType.HCL,
                                ZeroTrustPrinciple.DATA_SOVEREIGNTY,
                                f"S3 bucket {block_name} created in non-compliant region: {region}",
                                "high",
                                f"resource.{block_type}.{block_name}",
                                "Data Sovereignty"
                            )
                        )
            
            elif 'google_storage_bucket' in block_type:
                for block_name, block_config in blocks.items():
                    location = block_config.get('location', '')
                    if location and location not in self.allowed_regions:
                        self.violations.append(
                            PolicyViolation(
                                PolicyType.HCL,
                                ZeroTrustPrinciple.DATA_SOVEREIGNTY,
                                f"GCS bucket {block_name} created in non-compliant location: {location}",
                                "high",
                                f"resource.{block_type}.{block_name}",
                                "Data Sovereignty"
                            )
                        )
    
    def _check_temporal_access_hcl(self, config: Dict[str, Any]) -> None:
        """Check HCL for temporal access constraints"""
        resource_blocks = config.get('resource', [])
        
        for block_type, blocks in resource_blocks.items():
            if 'aws_iam_role' in block_type:
                for block_name, block_config in blocks.items():
                    # Check for maximum session duration settings
                    max_session_duration = block_config.get('max_session_duration', 3600)
                    if max_session_duration > 3600:  # More than 1 hour
                        self.violations.append(
                            PolicyViolation(
                                PolicyType.HCL,
                                ZeroTrustPrinciple.TEMPORAL_ACCESS,
                                f"IAM role {block_name} allows sessions longer than 1 hour: {max_session_duration}s",
                                "medium",
                                f"resource.{block_type}.{block_name}",
                                "Temporal Access"
                            )
                        )
    
    def _check_secret_management_hcl(self, code: str, config: Dict[str, Any]) -> None:
        """Check HCL for secret management issues"""
        # Look for potential secrets in the code
        for pattern in self.secret_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                # Avoid flagging common non-secret patterns
                if not self._is_false_positive(match.group()):
                    self.violations.append(
                        PolicyViolation(
                            PolicyType.HCL,
                            ZeroTrustPrinciple.SECRET_MANAGEMENT,
                            f"Potential hardcoded secret detected: {match.group()[:20]}...",
                            "high",
                            "code",
                            "Secret Management"
                        )
                    )
        
        # Check for sensitive values in variables
        variables = config.get('variable', {})
        for var_name, var_config in variables.items():
            if any(keyword in var_name.lower() for keyword in ['secret', 'password', 'token', 'key']):
                default = var_config.get('default', '')
                if default and not isinstance(default, dict) and str(default).strip():
                    self.violations.append(
                        PolicyViolation(
                            PolicyType.HCL,
                            ZeroTrustPrinciple.SECRET_MANAGEMENT,
                            f"Sensitive variable {var_name} has a hardcoded default value",
                            "high",
                            f"variable.{var_name}",
                            "Secret Management"
                        )
                    )
    
    def _check_auditing_immutability_hcl(self, config: Dict[str, Any]) -> None:
        """Check HCL for auditing and immutability settings"""
        resource_blocks = config.get('resource', [])
        
        for block_type, blocks in resource_blocks.items():
            if 'aws_s3_bucket' in block_type:
                for block_name, block_config in blocks.items():
                    # Check for versioning (related to immutability)
                    if not block_config.get('versioning', {}).get('enabled', False):
                        self.violations.append(
                            PolicyViolation(
                                PolicyType.HCL,
                                ZeroTrustPrinciple.AUDITING_IMMUTABILITY,
                                f"S3 bucket {block_name} does not have versioning enabled",
                                "medium",
                                f"resource.{block_type}.{block_name}",
                                "Auditing & Immutability"
                            )
                        )
                    
                    # Check for logging
                    if not block_config.get('logging', {}):
                        self.violations.append(
                            PolicyViolation(
                                PolicyType.HCL,
                                ZeroTrustPrinciple.AUDITING_IMMUTABILITY,
                                f"S3 bucket {block_name} does not have access logging configured",
                                "medium",
                                f"resource.{block_type}.{block_name}",
                                "Auditing & Immutability"
                            )
                        )
            
            elif 'aws_cloudtrail' in block_type:
                for block_name, block_config in blocks.items():
                    # Check if CloudTrail is enabled for multi-region
                    if not block_config.get('is_multi_region_trail', False):
                        self.violations.append(
                            PolicyViolation(
                                PolicyType.HCL,
                                ZeroTrustPrinciple.AUDITING_IMMUTABILITY,
                                f"CloudTrail {block_name} is not multi-region",
                                "medium",
                                f"resource.{block_type}.{block_name}",
                                "Auditing & Immutability"
                            )
                        )
    
    def _is_overly_permissive_iam(self, policy_json: Dict[str, Any]) -> bool:
        """Check if IAM policy is overly permissive"""
        statements = policy_json.get('Statement', [])
        
        for statement in statements:
            effect = statement.get('Effect', 'Allow')
            action = statement.get('Action', [])
            resource = statement.get('Resource', [])
            
            # Check for wildcard actions and resources
            if (effect == 'Allow' and 
                ('*' in action or any('*' in str(a) for a in action)) and 
                ('*' in resource or any('*' in str(r) for r in resource))):
                return True
        
        return False
    
    def _is_false_positive(self, candidate: str) -> bool:
        """Check if a detected secret is likely a false positive"""
        # Common non-secret patterns that might match our regex
        false_positives = [
            'example', 'test', 'mock', 'dummy', 'placeholder',
            'AAAA', 'BBBB', 'CCCC',  # Common placeholder patterns
        ]
        
        candidate_lower = candidate.lower()
        return any(fp in candidate_lower for fp in false_positives)
    
    def _analyze_acp(self, code: str) -> None:
        """Analyze ACP policy for Zero Trust compliance"""
        try:
            policy = json.loads(code)
            
            # Check for proper authentication
            self._check_acp_auth(policy)
            
            # Check for least privilege access
            self._check_acp_privileges(policy)
            
            # New aspect checks
            self._check_identity_centric_acp(policy)
            self._check_temporal_access_acp(policy)
            
            # Use LLM for more complex analysis
            self._llm_analysis(code, PolicyType.ACP)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid ACP policy: {str(e)}")
    
    def _check_acp_auth(self, policy: Dict[str, Any]) -> None:
        """Check ACP policy for authentication requirements"""
        # ACP-specific authentication checks
        if 'authentication' not in policy:
            self.violations.append(
                PolicyViolation(
                    PolicyType.ACP,
                    ZeroTrustPrinciple.EXPLICIT_VERIFICATION,
                    "No authentication mechanism defined",
                    "high",
                    "authentication",
                    "Authentication"
                )
            )
    
    def _check_acp_privileges(self, policy: Dict[str, Any]) -> None:
        """Check ACP policy for least privilege violations"""
        # Check for overly permissive rules
        rules = policy.get('rules', [])
        
        for i, rule in enumerate(rules):
            if rule.get('action', 'deny') == 'allow' and not rule.get('constraints'):
                self.violations.append(
                    PolicyViolation(
                        PolicyType.ACP,
                        ZeroTrustPrinciple.LEAST_PRIVILEGE,
                        f"Allow rule without constraints at index {i}",
                        "medium",
                        f"rules[{i}]",
                        "Least Privilege"
                    )
                )
    
    def _check_identity_centric_acp(self, policy: Dict[str, Any]) -> None:
        """Check ACP for identity-centric policy binding"""
        rules = policy.get('rules', [])
        
        for i, rule in enumerate(rules):
            subjects = rule.get('subjects', [])
            # Check if rules specify specific users rather than groups
            if not any('user:' in subject for subject in subjects) and subjects:
                self.violations.append(
                    PolicyViolation(
                        PolicyType.ACP,
                        ZeroTrustPrinciple.IDENTITY_CENTRIC,
                        f"Rule at index {i} uses group-based access rather than individual user identities",
                        "medium",
                        f"rules[{i}].subjects",
                        "Identity-Centric Policy"
                    )
                )
    
    def _check_temporal_access_acp(self, policy: Dict[str, Any]) -> None:
        """Check ACP for temporal access constraints"""
        rules = policy.get('rules', [])
        
        for i, rule in enumerate(rules):
            constraints = rule.get('constraints', {})
            # Check for time-based constraints
            if not any(key in constraints for key in ['time', 'date', 'expires']):
                self.violations.append(
                    PolicyViolation(
                        PolicyType.ACP,
                        ZeroTrustPrinciple.TEMPORAL_ACCESS,
                        f"Rule at index {i} has no temporal constraints",
                        "low",
                        f"rules[{i}].constraints",
                        "Temporal Access"
                    )
                )
    
    def _analyze_mcp(self, code: str) -> None:
        """Analyze MCP policy for Zero Trust compliance"""
        try:
            policy = json.loads(code)
            
            # Check for proper authentication
            self._check_mcp_auth(policy)
            
            # Check for device trust requirements
            self._check_mcp_device_trust(policy)
            
            # New aspect checks
            self._check_identity_centric_mcp(policy)
            self._check_temporal_access_mcp(policy)
            self._check_auditing_mcp(policy)
            
            # Use LLM for more complex analysis
            self._llm_analysis(code, PolicyType.MCP)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid MCP policy: {str(e)}")
    
    def _check_mcp_auth(self, policy: Dict[str, Any]) -> None:
        """Check MCP policy for authentication requirements"""
        # MCP-specific authentication checks
        if 'identity' not in policy:
            self.violations.append(
                PolicyViolation(
                    PolicyType.MCP,
                    ZeroTrustPrinciple.EXPLICIT_VERIFICATION,
                    "No identity verification mechanism defined",
                    "high",
                    "identity",
                    "Authentication"
                )
            )
    
    def _check_mcp_device_trust(self, policy: Dict[str, Any]) -> None:
        """Check MCP policy for device trust requirements"""
        # Check for device health requirements
        device_checks = policy.get('device_health', {})
        
        if not device_checks.get('os_version_check', False):
            self.violations.append(
                PolicyViolation(
                    PolicyType.MCP,
                    ZeroTrustPrinciple.DEVICE_TRUST,
                    "No OS version compliance check required",
                    "medium",
                    "device_health.os_version_check",
                    "Device Trust"
                )
            )
    
    def _check_identity_centric_mcp(self, policy: Dict[str, Any]) -> None:
        """Check MCP for identity-centric policy binding"""
        conditions = policy.get('conditions', {})
        
        # Check for MFA requirements
        if not conditions.get('mfa_required', False):
            self.violations.append(
                PolicyViolation(
                    PolicyType.MCP,
                    ZeroTrustPrinciple.IDENTITY_CENTRIC,
                    "No multi-factor authentication requirement specified",
                    "high",
                    "conditions.mfa_required",
                    "Identity-Centric Policy"
                )
            )
    
    def _check_temporal_access_mcp(self, policy: Dict[str, Any]) -> None:
        """Check MCP for temporal access constraints"""
        conditions = policy.get('conditions', {})
        
        # Check for session expiration
        if not conditions.get('max_session_duration', 0):
            self.violations.append(
                PolicyViolation(
                    PolicyType.MCP,
                    ZeroTrustPrinciple.TEMPORAL_ACCESS,
                    "No maximum session duration specified",
                    "medium",
                    "conditions.max_session_duration",
                    "Temporal Access"
                )
            )
    
    def _check_auditing_mcp(self, policy: Dict[str, Any]) -> None:
        """Check MCP for auditing capabilities"""
        # Check if audit logging is enabled
        if not policy.get('audit_logging', {}).get('enabled', False):
            self.violations.append(
                PolicyViolation(
                    PolicyType.MCP,
                    ZeroTrustPrinciple.AUDITING_IMMUTABILITY,
                    "Audit logging is not enabled",
                    "medium",
                    "audit_logging.enabled",
                    "Auditing & Immutability"
                )
            )
    
    def _llm_analysis(self, code: str, policy_type: PolicyType) -> None:
        """
        Use LLM for advanced analysis of policy code
        This is a placeholder for actual LLM integration
        """
        # In a real implementation, this would send the code to an LLM
        # with prompts tailored to Zero Trust principles and the policy type
        
        logger.info(f"Performing LLM analysis on {policy_type.value} code")
        
        # Example of what an LLM prompt might look like
        prompt = f"""
        Analyze the following {policy_type.value} code for Zero Trust security compliance.
        Focus on these principles:
        1. Least privilege access
        2. Explicit verification
        3. Assume breach mindset
        4. Microsegmentation
        5. Device and user trust
        6. Data encryption
        7. Identity-centric policy binding
        8. Data sovereignty and geolocation
        9. Temporal and contextual access constraints
        10. Secret management and credential hygiene
        11. Logging, monitoring, and immutability
        
        Code:
        {code}
        
        Identify any violations and return them in JSON format with:
        - principle: which Zero Trust principle is violated
        - message: description of the violation
        - severity: high, medium, or low
        - location: where in the code the violation occurs
        - aspect: which of the 5 main aspects this relates to
        """
        
        # In practice, you would call your LLM API here
        # response = self.llm_client.complete(prompt)
        # Parse the response and add violations to self.violations
        
        # For demonstration, we'll simulate an LLM finding
        if policy_type == PolicyType.OPENAPI and "http:" in code and "localhost" not in code:
            self.violations.append(
                PolicyViolation(
                    policy_type,
                    ZeroTrustPrinciple.DATA_ENCRYPTION,
                    "LLM detected unencrypted HTTP communication in API",
                    "high",
                    "servers",
                    "Data Encryption"
                )
            )

# @title Utility Functions
def detect_policy_type(code: str) -> PolicyType:
    """Detect the type of policy based on code content"""
    code = code.strip()
    
    # Check for OpenAPI
    if code.startswith('openapi:') or code.startswith('{"openapi":') or code.startswith('---'):
        return PolicyType.OPENAPI
    
    # Check for ACP
    if '"acpPolicy"' in code or 'acp_policy' in code:
        return PolicyType.ACP
    
    # Check for MCP
    if '"mcpPolicy"' in code or 'mcp_policy' in code:
        return PolicyType.MCP
    
    # Check for HCL
    if 'resource "' in code or 'provider "' in code:
        return PolicyType.HCL
    
    return PolicyType.UNKNOWN

def generate_summary_report(violations: List[PolicyViolation]) -> Dict[str, Any]:
    """Generate a comprehensive summary report of all violations"""
    summary = {
        "total_violations": len(violations),
        "by_severity": {"high": 0, "medium": 0, "low": 0},
        "by_aspect": {},
        "by_policy_type": {},
        "by_principle": {}
    }
    
    for violation in violations:
        # Count by severity
        summary["by_severity"][violation.severity] += 1
        
        # Count by aspect
        aspect = violation.aspect
        summary["by_aspect"][aspect] = summary["by_aspect"].get(aspect, 0) + 1
        
        # Count by policy type
        policy_type = violation.policy_type.value
        summary["by_policy_type"][policy_type] = summary["by_policy_type"].get(policy_type, 0) + 1
        
        # Count by principle
        principle = violation.principle.value
        summary["by_principle"][principle] = summary["by_principle"].get(principle, 0) + 1
    
    return summary

def print_detailed_report(violations: List[PolicyViolation], summary: Dict[str, Any]):
    """Print a detailed report of the analysis"""
    display(Markdown("## ZERO TRUST CODE GUARDRAILS ANALYSIS REPORT"))
    display(Markdown("---"))
    
    display(Markdown(f"**Total violations found:** {summary['total_violations']}"))
    display(Markdown(f"**High severity:** {summary['by_severity']['high']}"))
    display(Markdown(f"**Medium severity:** {summary['by_severity']['medium']}"))
    display(Markdown(f"**Low severity:** {summary['by_severity']['low']}"))
    
    display(Markdown("### Violations by aspect:"))
    for aspect, count in summary['by_aspect'].items():
        display(Markdown(f"  - **{aspect}:** {count}"))
    
    display(Markdown("### Violations by policy type:"))
    for policy_type, count in summary['by_policy_type'].items():
        display(Markdown(f"  - **{policy_type}:** {count}"))
    
    display(Markdown("### Violations by principle:"))
    for principle, count in summary['by_principle'].items():
        display(Markdown(f"  - **{principle}:** {count}"))
    
    display(Markdown("### DETAILED VIOLATIONS:"))
    display(Markdown("---"))
    
    # Group violations by aspect for better readability
    violations_by_aspect = {}
    for violation in violations:
        if violation.aspect not in violations_by_aspect:
            violations_by_aspect[violation.aspect] = []
        violations_by_aspect[violation.aspect].append(violation)
    
    for aspect, aspect_violations in violations_by_aspect.items():
        display(Markdown(f"#### {aspect.upper()} VIOLATIONS:"))
        for violation in aspect_violations:
            display(Markdown(f"- {violation}"))

# @title Example Code Snippets
# @markdown These are the code examples that will be analyzed

openapi_example = """
openapi: 3.0.0
info:
  title: Example API
  version: 1.0.0
servers:
  - url: http://api.example.com  # Violation: HTTP instead of HTTPS
  - url: https://api-us.example.com  # OK: HTTPS
paths:
  /users:
    get:
      summary: Get all users
      responses:
        '200':
          description: OK
  /admin:
    post:
      summary: Admin operation
      security:
        - oauth2: []  # Violation: No scopes specified
      responses:
        '200':
          description: OK
components:
  securitySchemes:
    oauth2:
      type: oauth2
      flows:
        implicit:
          authorizationUrl: https://example.com/oauth/authorize
          scopes:
            read: read access
            write: write access
"""

hcl_example = """
resource "aws_security_group" "example" {
  name        = "example"
  description = "Example security group"
  
  ingress {
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Violation: Overly permissive
  }
}

resource "aws_s3_bucket" "data" {
  bucket = "my-data-bucket"
  region = "us-west-2"  # Violation: Not in allowed region
  
  versioning {
    enabled = false  # Violation: Versioning disabled
  }
}

variable "db_password" {
  description = "Database password"
  default = "secret123"  # Violation: Hardcoded secret
}
"""

acp_example = """
{
  "acpPolicy": {
    "name": "example-policy",
    "rules": [
      {
        "action": "allow",
        "subjects": ["group:admin"],
        "constraints": {}  # Violation: No constraints
      }
    ]
  }
}
"""

# @title Run Analysis
# @markdown Execute this cell to run the Zero Trust analysis on the example code snippets

# Initialize the guardrail engine with allowed regions
engine = MultiModalGuardrailEngine(allowed_regions=["EU", "europe-west1", "europe-west2"])

all_violations = []

# Analyze OpenAPI example
openapi_type = detect_policy_type(openapi_example)
openapi_violations = engine.analyze_code(openapi_example, openapi_type)
all_violations.extend(openapi_violations)

# Analyze HCL example
hcl_type = detect_policy_type(hcl_example)
hcl_violations = engine.analyze_code(hcl_example, hcl_type)
all_violations.extend(hcl_violations)

# Analyze ACP example
acp_type = detect_policy_type(acp_example)
acp_violations = engine.analyze_code(acp_example, acp_type)
all_violations.extend(acp_violations)

# Generate and print report
summary = generate_summary_report(all_violations)
print_detailed_report(all_violations, summary)

# Print recommendations
display(Markdown("## RECOMMENDATIONS:"))
display(Markdown("---"))

if summary['by_severity']['high'] > 0:
    display(Markdown("❌ **CRITICAL:** Address high severity issues immediately"))
else:
    display(Markdown("✅ No critical issues found"))

if summary['by_aspect'].get('Secret Management', 0) > 0:
    display(Markdown("🔒 **Review all potential secret leaks and use secure secret management**"))

if summary['by_aspect'].get('Data Sovereignty', 0) > 0:
    display(Markdown("🌍 **Ensure data storage complies with regional regulations**"))

if summary['by_aspect'].get('Identity-Centric Policy', 0) > 0:
    display(Markdown("👤 **Implement identity-centric access controls with proper scoping**"))

if summary['total_violations'] == 0:
    display(Markdown("🎉 **Excellent! No Zero Trust violations detected**"))

# @title Test with Your Own Code
# @markdown Paste your own code in the text area below and run the analysis

code_type = "HCL"  # @param ["OpenAPI", "HCL", "ACP", "MCP"]
your_code = '''# Paste your code here
resource "aws_s3_bucket" "example" {
  bucket = "my-example-bucket"
  region = "us-west-2"
  
  versioning {
    enabled = true
  }
}
'''

# @markdown Run the analysis on your custom code
def analyze_custom_code(code, code_type):
    """Analyze custom code provided by the user"""
    policy_type = PolicyType[code_type.upper()]
    engine = MultiModalGuardrailEngine(allowed_regions=["EU", "europe-west1", "europe-west2"])
    violations = engine.analyze_code(code, policy_type)
    
    display(Markdown(f"## Analysis of Your {code_type} Code"))
    display(Markdown("---"))
    
    if violations:
        summary = generate_summary_report(violations)
        display(Markdown(f"**Found {len(violations)} violations**"))
        
        for violation in violations:
            display(Markdown(f"- **{violation.severity.upper()}**: {violation.message}"))
    else:
        display(Markdown("✅ **No violations found! Your code complies with Zero Trust principles.**"))
    
    return violations

# Run analysis on custom code
custom_violations = analyze_custom_code(your_code, code_type)
'''
How to Use This Notebook
Run the Setup: Execute the first cell to install dependencies and import libraries

Review the Code: The notebook defines all necessary classes and functions for Zero Trust analysis

Run the Analysis: Execute the "Run Analysis" cell to see the Zero Trust guardrails in action

Test Your Own Code: Use the last cell to test your own OpenAPI, HCL, ACP, or MCP code

Key Features
Multi-Modal Support: Analyzes OpenAPI specs, ACP/MCP policies, and HCL configurations

Zero Trust Principles Enforcement:

Least privilege access

Explicit verification

Assume breach mindset

Microsegmentation

Device and user trust

Data encryption

Identity-centric policy binding

Data sovereignty and geolocation

Temporal and contextual access constraints

Secret management and credential hygiene

Logging, monitoring, and immutability

Comprehensive Reporting: Detailed violation reports with severity levels and recommendations

Custom Code Testing: Ability to test your own infrastructure code against Zero Trust principles

This Colab notebook provides a complete environment for experimenting with Zero Trust code guardrails and understanding how multi-modal LLMs can enhance security policy enforcement.
'''
